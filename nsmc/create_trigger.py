import os
import sys
import argparse

import wandb
import torch

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nsmc import attacks
from nsmc.utils import (
    seed_everything,
    load_data,
    tokenized_dataset,
    SentimentClassificationDataset,
    add_hooks,
    get_embedding_weight,
    get_accuracy,
    get_average_grad,
    get_best_candidates,
    save_pickle_file,
    get_sentiment_token_index_set,
)


# gpu setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set the GPU 0 to use


def main(args):
    # wandb setting
    wandb_config = {
        'approach': 'hot-flip',
        'dataset_label': args.dataset_label_filter,
    }

    wandb.init(project=args.project_name,
               name=f'{"positive to negative" if args.dataset_label_filter else "negative to positive"}-trigger-{args.num_trigger_tokens}',
               config=wandb_config,
               reinit=True,
               )

    # seed 고정
    seed_everything(args.seed)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    # model_config load
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    model_config.num_labels = args.num_labels

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    word_to_index_dict = tokenizer.vocab
    index_to_word_dict = {v: k for k, v in tokenizer.vocab.items()}
    vocab_size = tokenizer.vocab_size

    # model load
    best_model_dir = os.path.join(best_model_path, args.model_name_or_path.split('/')[-1], args.run_name)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    model.train().to(device)

    add_hooks(model, vocab_size)  # add gradient hooks to embeddings
    embedding_weight = get_embedding_weight(model, vocab_size)  # save the word embedding matrix

    # load NSMC dataset
    train_dataset = load_data(dataset_dir=os.path.join(data_path, 'ratings_train.txt'))

    # train / valid set split
    train_df, valid_df = train_test_split(train_dataset, test_size=0.2, shuffle=True, stratify=train_dataset['label'],
                                          random_state=args.seed)

    # filter the dataset to only positive or negative examples
    # (the trigger will cause the opposite prediction)
    dataset_label_filter = args.dataset_label_filter
    targeted_valid_df = valid_df[valid_df['label'] == dataset_label_filter]
    print('postive to negative' if args.dataset_label_filter == 1 else 'negative to positive',
          f'targeted_valid_df 개수 : {len(targeted_valid_df)}')

    # get sentiment token set (args.dataset_label_filter == 0 이면 긍정 감성 단어 1이면 부정 감성 단어)
    sentiment_token_index_set, remove_token_index_set = get_sentiment_token_index_set(args, data_path)

    # valid label setting
    targeted_valid_label = targeted_valid_df['label'].tolist()

    # tokenizing dataset
    tokenized_targeted_valid = tokenized_dataset(targeted_valid_df, tokenizer, args)

    # make dataset
    targeted_valid_dataset = SentimentClassificationDataset(tokenized_targeted_valid, targeted_valid_label)

    # get accuracy before adding triggers
    get_accuracy(args, device, model, targeted_valid_dataset, index_to_word_dict, trigger_token_ids=None)
    model.train()

    # initialize triggers which are concatenated to the input
    trigger_token_ids = [word_to_index_dict.get("the")] * args.num_trigger_tokens

    # sample batches, update the triggers, and repeat
    trigger_accuracy_dict = {}
    wandb_table = wandb.Table(columns=['trigger', 'attack_success_rate'])
    for epoch in range(args.epoch):
        print('*' * 50)
        print(f'{epoch + 1} / {args.epoch} Epochs')
        print('*' * 50)
        for batch in tqdm(DataLoader(targeted_valid_dataset, batch_size=args.universal_perturb_batch_size,
                                     shuffle=True)):
            # get accuracy with current triggers
            accuracy, trigger = get_accuracy(args, device, model, targeted_valid_dataset, index_to_word_dict,
                                             trigger_token_ids)
            trigger_accuracy_dict[trigger] = accuracy
            wandb_table.add_data(trigger, round(1 - accuracy, 4))
            model.train()

            # get gradient w.r.t. trigger embeddings for current batch
            averaged_grad = get_average_grad(args, device, model, batch, trigger_token_ids)

            # pass the gradients to a particular attack to generate token candidates for each token.
            # cand_trigger_token_ids = attacks.hotflip_attack(averaged_grad,
            #                                                 embedding_weight,
            #                                                 trigger_token_ids,
            #                                                 num_candidates=args.num_candidates,
            #                                                 increase_loss=True,
            #                                                 )
            cand_trigger_token_ids = attacks.hotflip_attack_without_sentiment_token(
                averaged_grad,
                embedding_weight,
                trigger_token_ids,
                sentiment_token_index_set,
                remove_token_index_set,
                num_candidates=args.num_candidates,
                increase_loss=True,
            )

            # Tries all of the candidates and returns the trigger sequence with highest loss.
            trigger_token_ids = get_best_candidates(args, device, model, batch, trigger_token_ids,
                                                    cand_trigger_token_ids)

    # print accuracy after adding triggers
    accuracy, trigger = get_accuracy(args, device, model, targeted_valid_dataset, index_to_word_dict, trigger_token_ids)
    trigger_accuracy_dict[trigger] = accuracy
    wandb_table.add_data(trigger, round(1 - accuracy, 4))
    wandb.log({f'{"positive to negative" if args.dataset_label_filter else "negative to positive"}'
               f'-trigger-{args.num_trigger_tokens}-trigger_attack_success_rate': wandb_table})

    filename = 'trigger_accuracy_{}word_{}_dict'.format(args.num_trigger_tokens,
                                                        'n2p' if args.dataset_label_filter == 0 else 'p2n')
    save_pickle_file(path=os.path.join(data_path, 'trigger_dict'), filename=filename, obj=trigger_accuracy_dict)


if __name__ == '__main__':
    base_path = os.path.dirname(os.path.dirname(__file__))
    best_model_path = os.path.join(base_path, 'best_model')
    prediction_path = os.path.join(base_path, 'prediction')
    data_path = os.path.join(base_path, 'data')

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epoch', type=int, default=3, help='epoch (default: 3)')
    parser.add_argument('--num_candidates', type=int, default=40, help='number of candidate (default: 40)')
    parser.add_argument('--num_trigger_tokens', type=int, default=3, help='number of trigger token  (default: 3)')
    parser.add_argument('--model_name_or_path', type=str, default='klue/roberta-base',
                        help='model_name_or_path used for training')
    parser.add_argument('--run_name', type=str, required=True,
                        help='run_name used for training')
    parser.add_argument('--universal_perturb_batch_size', type=int, default=80,
                        help='batch size per device during evaluation (default: 80)')
    parser.add_argument('--num_labels', type=int, default=2, help='number of labels (default: 2)')
    parser.add_argument('--dataset_label_filter', type=int, default=1, help='label filter (negative: 0 positive: 1)')
    parser.add_argument('--project_name', type=str, default='tmax ojt create trigger',
                        help='wandb project name (default: tmax_ojt)')

    args = parser.parse_args()

    print(args)
    main(args)
