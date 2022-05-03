import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import (
    load_data,
    tokenized_dataset,
    SentimentClassificationDataset,
)


# gpu setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU 0 to use


def inference(test_df, test_label, args):
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

    # model load
    best_model_dir = os.path.join(best_model_path, args.model_name_or_path.split('/')[-1], args.run_name)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    model.to(device)

    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_df, tokenizer, args)

    # make dataset
    test_dataset = SentimentClassificationDataset(tokenized_test, test_label)

    # make dataloader
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    model.eval()

    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(test_loader)):
        # 모델이 roberta 인 경우 token_type_ids 제거
        if 'roberta' in args.model_name_or_path:
            data = {k: v.to(device) for k, v in data.items() if k not in ['labels', 'token_type_ids']}
        else:
            data = {k: v.to(device) for k, v in data.items() if k not in ['labels']}

        with torch.no_grad():
            # 모델에 데이터를 넣고 추론
            outputs = model(**data)

        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()


def main(args):
    # load test dataset
    test_df = load_data(dataset_dir=os.path.join(data_path, 'ratings_test.txt'), is_train=False)
    test_label = test_df['label'].values

    output_pred, output_prob = inference(test_df, test_label, args)

    accuracy = round(accuracy_score(test_label, output_pred), 2)

    # prediction dataframe
    prediction = pd.DataFrame({'id': test_df['id'].tolist(), 'pred_label': output_pred, 'pred_prob': output_prob,
                               'label': test_label})

    prediction_dir = os.path.join(prediction_path, args.model_name_or_path.split('/')[-1], args.run_name)
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)

    prediction_dir = os.path.join(prediction_dir, f'{accuracy}_prediction.csv')
    prediction.to_csv(prediction_dir, index=False)


if __name__ == '__main__':
    base_path = os.path.dirname(os.path.dirname(__file__))
    best_model_path = os.path.join(base_path, 'best_model')
    prediction_path = os.path.join(base_path, 'prediction')
    data_path = os.path.join(base_path, 'data')

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, default='klue/roberta-base',
                        help='model_name_or_path used for training')
    parser.add_argument('--run_name', type=str, required=True,
                        help='run_name used for training')
    parser.add_argument('--eval_batch_size', type=int, default=256,
                        help='batch size per device during evaluation (default: 256)')
    parser.add_argument('--prediction_dir', type=str, default=prediction_path,
                        help='predict result folder')
    parser.add_argument('--num_labels', type=int, default=2, help='number of labels (default: 2)')

    args = parser.parse_args()

    print(args)
    main(args)
