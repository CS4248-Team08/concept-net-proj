import pickle
import random
from itertools import cycle
import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, DistilBertTokenizer


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

def get_dataloaders(train_test_split_frac, batch_size):
    sampled_problems = pickle.load(open(
            '../../data/science/paths.pkl', 'rb'), encoding='latin1')

    texts = dict()
    print('loading problem plain texts')
    for id_num in sampled_problems:
        f_short = sampled_problems[id_num]['forward']['short']
        r_short = sampled_problems[id_num]['reverse']['short']
        texts[id_num+'f'] = f_short
        texts[id_num+'r'] = r_short

    print('loading labeled pairs')
    all_pairs = []  # list of id tuples (good, bad)
    for line in open('../../data/science/answers.txt'):
        first, second, good = line.strip().split('_')
        if first == good:
            bad = second
        elif second == good:
            bad = first
        g_len = (len(texts[good].strip().split(' '))+1)/2
        b_len = (len(texts[bad].strip().split(' '))+1)/2
        if g_len != 4 or b_len != 4:
            continue
        all_pairs.append((good, bad))

    random.shuffle(all_pairs)

    # all_pairs = all_pairs[:1050]

    split = int(train_test_split_frac * len(all_pairs))
    train_pairs = all_pairs[:split]
    test_pairs = all_pairs[split:]

    train_df = augment_data(train_pairs, texts)
    test_df = get_test_df(test_pairs, texts)

    train = DataLoader(BertDataset(train_df), batch_size=batch_size, shuffle=True)
    test = DataLoader(BertDataset(test_df), batch_size=batch_size, shuffle=True)

    return train, test

def process_short(text):
    p = re.sub(r"[<->]{2,3}", " ", text)  # remove arrows
    return p

def augment_data(train_pairs, texts):
    data = []
    labels = []
    for good, bad in train_pairs:
        data += [(texts[good], texts[bad]), (texts[bad], texts[good])]
        labels += [1, 0]

    df = pd.DataFrame({"texts": data, "labels": labels})
    # return df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df  # DataLoader will do shuffle

def get_test_df(test_pairs, texts):
    data = []
    labels = []
    for good, bad in test_pairs:
        if random.random() > 0.5:
            data.append((texts[good], texts[bad]))
            labels.append(1)
        else:
            data.append((texts[bad], texts[good]))
            labels.append(0)

    df = pd.DataFrame({"texts": data, "labels": labels})
    # return df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df  # DataLoader will do shuffle

class BertDataset(Dataset):

    def __init__(self, df):
        self.labels = torch.tensor(df['labels'])
        # self.data = []
        # for path1, path2 in df['texts']:
        #     path1_id = tokenizer.encode(path1, add_special_tokens = False)
        #     path2_id = tokenizer.encode(path2, add_special_tokens = False)
        #     pair_token_ids = [tokenizer.cls_token_id] + path1_id + [tokenizer.sep_token_id] + path2_id + [tokenizer.sep_token_id]
        #     path1_len = len(path1_id)
        #     path2_len = len(path2_id)

        #     segment_ids = torch.tensor([0] * (path1_len + 2) + [1] * (path2_len + 1))  # sentence 0 and sentence 1
        #     attention_mask_ids = torch.tensor([1] * (path1_len + path2_len + 3))  # mask padded values

        #     self.data.append({
        #         "input_ids": pair_token_ids,
        #         "token_type_ids": segment_ids,
        #         "attention_mask": attention_mask_ids
        #     })

        self.data = [
            tokenizer(
                text=t1,
                text_pair=t2,
                padding='max_length',
                max_length = 100,
                truncation=True,
                return_tensors="pt"
            )
            for t1, t2 in df['texts']
        ]
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
