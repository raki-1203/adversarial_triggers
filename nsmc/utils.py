import heapq
import random
import numpy as np
import pandas as pd
import re

import torch
import torch.nn as nn
import torch.optim as optim

from glob import glob
from pathlib import Path

from tqdm import tqdm
from copy import deepcopy
from operator import itemgetter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score


# returns the wordpiece embedding weight matrix
def get_embedding_weight(model, vocab_size):
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == vocab_size:  # only add a hook to wordpiece embeddings, not position embeddings
                return module.weight.detach()


# add hooks for embeddings
def add_hooks(model, vocab_size):
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == vocab_size:  # only add a hook to wordpiece embeddings, not position
                module.weight.requires_grad = True
                module.register_full_backward_hook(extract_grad_hook)


# hook used in add_hooks()
extracted_grads = []


def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])


def get_average_grad(args, device, model, batch, trigger_token_ids, target_label=None):
    """
    Computes the average gradient w.r.t. the trigger tokens when prepended to every example
    in the batch. If target_label is set, that is used as the ground-truth label.
    """
    # create an dummy optimizer for backprop
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()

    # prepend triggers to the batch
    original_labels = batch['labels'].clone()
    if target_label is not None:
        # set the labels equal to the target (backprop from the target class, not model prediction)
        batch['labels'] = int(target_label) * torch.ones_like(batch['labels']).cuda()
    global extracted_grads
    extracted_grads = []  # clear existing stored grads
    _, loss = evaluate_batch(args, device, model, batch, trigger_token_ids)
    loss.backward()
    # index 0 has the hypothesis grads for SNLI. For SST, the list is of size 1.
    grads = extracted_grads[0].cpu()
    batch['labels'] = original_labels  # reset labels

    # average grad across batch size, result only makes sense for trigger tokens at the front
    averaged_grad = torch.sum(grads, dim=0)
    averaged_grad = averaged_grad[0:len(trigger_token_ids)]  # return just trigger grads
    return averaged_grad


def get_best_candidates(args, device, model, batch, trigger_token_ids, cand_trigger_token_ids, beam_size=1):
    """"
    Given the list of candidate trigger token ids (of number of trigger words by number of candidates
    per word), it finds the best new candidate trigger.
    This performs beam search in a left to right fashion.
    """
    # first round, no beams, just get the loss for each of the candidates in index 0.
    # (indices 1-end are just the old trigger)
    loss_per_candidate = get_loss_per_candidate(args, device, 0, model, batch, trigger_token_ids,
                                                cand_trigger_token_ids)
    # maximize the loss
    top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))

    # top_candidates now contains beam_size trigger sequences, each with a different 0th token
    for idx in range(1, len(trigger_token_ids)):  # for all trigger tokens, skipping the 0th (we did it above)
        loss_per_candidate = []
        for cand, _ in top_candidates:  # for all the beams, try all the candidates at idx
            loss_per_candidate.extend(get_loss_per_candidate(args, device, idx, model, batch, cand,
                                                             cand_trigger_token_ids))
        top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))
    return max(top_candidates, key=itemgetter(1))[0]


def get_loss_per_candidate(args, device, index, model, batch, trigger_token_ids, cand_trigger_token_ids):
    """
    For a particular index, the function tries all of the candidate tokens for that index.
    The function returns a list containing the candidate triggers it tried, along with their loss.
    """
    if isinstance(cand_trigger_token_ids[0], (np.int64, int)):
        print("Only 1 candidate for index detected, not searching")
        return trigger_token_ids
    loss_per_candidate = []
    # loss for the trigger without trying the candidates
    _, curr_loss = evaluate_batch(args, device, model, batch, trigger_token_ids)
    curr_loss = curr_loss.cpu().detach().numpy()
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
    for cand_id in range(len(cand_trigger_token_ids[0])):
        trigger_token_ids_one_replaced = deepcopy(trigger_token_ids)  # copy trigger
        trigger_token_ids_one_replaced[index] = cand_trigger_token_ids[index][cand_id]  # replace one token
        _, loss = evaluate_batch(args, device, model, batch, trigger_token_ids_one_replaced)
        loss = loss.cpu().detach().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    return loss_per_candidate


def get_accuracy(args, device, model, valid_dataset, index_to_word_dict, trigger_token_ids=None):
    """
    When trigger_token_ids is None, gets accuracy on the dev_dataset. Otherwise, gets accuracy with
    triggers prepended for the whole dev_dataset.
    """
    model.eval()  # model should be in eval() already, but just in case
    labels = []
    preds = []
    if trigger_token_ids is None:
        for batch in tqdm(DataLoader(valid_dataset, batch_size=args.universal_perturb_batch_size, shuffle=False)):
            pred = evaluate_batch(args, device, model, batch, trigger_token_ids)

            labels.append(batch['labels'])
            preds.append(pred)

        preds = np.concatenate(preds).tolist()
        labels = np.concatenate(labels).tolist()
        accuracy = round(accuracy_score(labels, preds), 4)
        print('*' * 50)
        print(f"Accuracy (Without Triggers): {accuracy}")
        print('*' * 50)
    else:
        print_string_list = []
        for idx in trigger_token_ids:
            print_string_list.append(index_to_word_dict.get(idx, '[UNK]'))
        print_string = ', '.join(print_string_list)

        for batch in tqdm(DataLoader(valid_dataset, batch_size=args.universal_perturb_batch_size, shuffle=False)):
            pred, _ = evaluate_batch(args, device, model, batch, trigger_token_ids)

            labels.append(batch['labels'])
            preds.append(pred)

        preds = np.concatenate(preds).tolist()
        labels = np.concatenate(labels).tolist()
        accuracy = round(accuracy_score(labels, preds), 4)
        print('*' * 50)
        print(f"Accuracy (Current Triggers: {print_string}): {accuracy}")
        print('*' * 50)


def evaluate_batch(args, device, model, batch, trigger_token_ids=None):
    """
    If trigger_token_ids is not None, then it will append the tokens to the input.
    This funtion is used to get the model's accuracy and/or the loss with/without the trigger.
    """
    # 모델이 roberta 인 경우 token_type_ids 제거
    if 'roberta' in args.model_name_or_path:
        data = {k: v.to(device) for k, v in batch.items() if k not in ['labels', 'token_type_ids']}
    else:
        data = {k: v.to(device) for k, v in batch.items() if k not in ['labels']}

    if trigger_token_ids is None:
        outputs = model(**data)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)
        return result
    else:
        trigger_sequence_tensor = torch.LongTensor(deepcopy(trigger_token_ids))
        trigger_sequence_tensor = trigger_sequence_tensor.repeat(len(data['input_ids']), 1).cuda()
        original_input_ids = data['input_ids'].clone()
        original_attention_mask = data['attention_mask'].clone()
        trigger_attention_mask = torch.ones_like(trigger_sequence_tensor)
        data['input_ids'] = torch.cat((trigger_sequence_tensor, original_input_ids), 1)
        data['attention_mask'] = torch.cat((trigger_attention_mask, original_attention_mask), 1)
        outputs = model(**data)

        logits = outputs[0]
        label = batch['labels'].to(device)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, label)
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)
        return result, loss


# 학습한 모델을 재생산하기 위해 seed를 고정
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def load_data(dataset_dir, is_train=True):
    """ txt 파일을 pandas Dataframe 으로 불러옵니다. """
    df = pd.read_csv(dataset_dir, sep='\t')
    total_row = df.shape[0]

    # NaN 값 제거
    na_row = df[df.document.isnull()].shape[0]
    df.dropna(inplace=True)
    print(f'NaN row 수: {na_row}')

    if is_train:
        duplicated_row = df.duplicated(subset=['document', 'label']).sum()
        df.drop_duplicates(subset=['document'], inplace=True)
        print(f'중복 row 수: {duplicated_row}')
    else:
        duplicated_row = 0

    print(f'남은 row 수 : {total_row - na_row - duplicated_row}')

    return df


def tokenized_dataset(dataset, tokenizer, args):
    tokenized_sentences = tokenizer(
        list(dataset['document']),
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False if 'roberta' in args.model_name_or_path else True,
    )

    return tokenized_sentences


class SentimentClassificationDataset(Dataset):

    def __init__(self, dataset, labels):
        self.pair_dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.pair_dataset['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # calculate accuracy using sklearn's function
    f1 = f1_score(labels, preds) * 100.0
    acc = accuracy_score(labels, preds)

    return {
        'f1': f1,
        'accuracy': acc,
    }


def get_token_index(tokenizer, word):
    return tokenizer.vocab.get(word, 3)




