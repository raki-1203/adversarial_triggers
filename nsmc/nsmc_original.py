import argparse
import sys
import os

import torch
import wandb as wandb

from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback, default_data_collator,
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import (
    seed_everything,
    load_data,
    tokenized_dataset,
    SentimentClassificationDataset,
    increment_path,
    compute_metrics,
)


# wandb description silent
os.environ['WANDB_SILENT'] = "true"

# gpu setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set the GPU 0 to use


def train(train_df, valid_df, train_label, valid_label, args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    # model_config load
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    model_config.num_labels = args.num_labels

    # tokenizer load
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # model load
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=model_config)
    model.eval()
    model.to(device)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_df, tokenizer, args)
    tokenized_valid = tokenized_dataset(valid_df, tokenizer, args)

    # make dataset
    train_dataset = SentimentClassificationDataset(tokenized_train, train_label)
    valid_dataset = SentimentClassificationDataset(tokenized_valid, valid_label)

    # increment_path 사용해서 동일한 이름의 폴더명일 경우 이름뒤에 2, 3, ... 식으로 번호 추가
    output_dir = increment_path(os.path.join(args.output_dir, args.run_name))

    # training argument setting
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.valid_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=args.load_best_model_at_end,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        report_to=args.report_to,
    )

    # Trainer setting
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=valid_dataset if training_args.do_eval else None,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    # train model
    trainer.train()
    save_dir = increment_path(os.path.join('./best_model', args.run_name))
    model.save_pretrained(save_dir)

    # 마지막 최종 best model 로 평가한 결과 저장해서 return
    eval_result = trainer.evaluate(valid_dataset)

    return eval_result


def main(args):
    # seed 고정
    seed_everything(args.seed)

    # load NSMC dataset
    train_dataset = load_data(dataset_dir=os.path.join(data_path, 'ratings_train.txt'))

    train_df, valid_df = train_test_split(train_dataset, test_size=0.2, shuffle=True, stratify=train_dataset['label'])

    # train, valid label setting
    train_label = train_df['label'].values
    valid_label = valid_df['label'].values

    # wandb setting
    wandb_config = {
        'model_name': args.model_name_or_path,
        'epochs': args.num_train_epochs,
        'train_batch_size': args.train_batch_size,
        'valid_batch_size': args.valid_batch_size,
        'learning_rate': args.learning_rate,
    }

    wandb.init(project=args.project_name,
               name=f'{args.run_name}',
               config=wandb_config,
               reinit=True,
               )

    result = train(train_df, valid_df, train_label, valid_label, args)

    # 최종 결과 출력
    print(f'f1 score: {result["eval_f1"]}')
    print(f'accuracy: {result["eval_accuracy"]}')


if __name__ == '__main__':
    base_path = os.path.dirname(os.path.dirname(__file__))
    output_path = os.path.join(base_path, 'output')
    log_path = os.path.join(base_path, 'logs')
    data_path = os.path.join(base_path, 'data')

    parser = argparse.ArgumentParser()

    # training arguments
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--do_train', type=bool, default=True, help='(default: True)')
    parser.add_argument('--do_eval', type=bool, default=True, help='(default: True)')
    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help='total number of training epochs (default: 10)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of updates steps to accumulate the gradients for, '
                             'before performing a backward/update pass. (default: 4)')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='batch size per device during training (default: 35)')
    parser.add_argument('--valid_batch_size', type=int, default=256, help='batch size for evaluation (default: 128)')
    parser.add_argument('--model_name_or_path', type=str, default='klue/roberta-base',
                        help='what kinds of models (default: klue/roberta-large)')
    parser.add_argument('--run_name', type=str, default='exp', help='name of the W&B run (default: exp)')

    # training arguments that don't change well
    parser.add_argument('--output_dir', type=str, default=output_path, help='output directory (default: ./output)')
    parser.add_argument('--save_total_limit', type=int, default=2, help='number of total save model (default: 2)')
    parser.add_argument('--save_steps', type=int, default=100, help='model saving step (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning_rate (default: 5e-5)')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='number of warmup steps for learning rate scheduler (default: 500)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='strength of weight decay (default: 0.01)')
    parser.add_argument('--logging_dir', type=str, default=log_path,
                        help='directory for storing logs (default: ./logs)')
    parser.add_argument('--logging_steps', type=int, default=100, help='log saving step (default: 100)')
    parser.add_argument('--evaluation_strategy', type=str, default='steps',
                        help='evaluation strategy to adopt during training (default: steps)')
    parser.add_argument('--eval_steps', type=int, default=100, help='evaluation step (default: 100)')
    parser.add_argument('--metric_for_best_model', type=str, default='accuracy',
                        help='metric_for_best_model (default: accuracy, f1)')
    parser.add_argument('--load_best_model_at_end', type=bool, default=True, help='(default: True)')
    parser.add_argument('--num_labels', type=int, default=2, help='number of labels (default: 2)')
    parser.add_argument('--report_to', type=str, default='wandb', help='(default: wandb)')
    parser.add_argument('--project_name', type=str, default='tmax ojt',
                        help='wandb project name (default: tmax_ojt)')
    parser.add_argument('--fp16', type=bool, default=True,
                        help='Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='number of early_stopping_patience (default: 5)')

    args = parser.parse_args()
    args.run_name = args.model_name_or_path.split('/')[-1] + '/' + args.run_name
    print(args)

    main(args)
