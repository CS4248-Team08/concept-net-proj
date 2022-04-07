
import os
import pickle

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import torch
from torch import nn
from torch.optim import Adam

from transformers import GPT2Model, GPT2Config, GPT2Tokenizer

from tqdm import tqdm

import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer

#creating tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({
  "eos_token": "</s>",
  "bos_token": "<s>",
  "unk_token": "<unk>",
  "pad_token": "<pad>",
  "mask_token": "<mask>"
})


config = GPT2Config(
  vocab_size=tokenizer.vocab_size,
  bos_token_id=tokenizer.bos_token_id,
  eos_token_id=tokenizer.eos_token_id
)

EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-6

class GPT2Classifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(GPT2Classifier, self).__init__()

        self.gpt = GPT2Model.from_pretrained('gpt2')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.activation = nn.Sigmoid()
    def forward(self, input_id, mask):
        _, x = self.gpt(input_ids= input_id, attention_mask=mask,return_dict=False)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, text_col='Text', label_col='Label'):
        self.labels = [float(x) for x in df[label_col]]
        self.texts = [
            tokenizer(
                text, 
                padding='max_length',
                max_length = 1024, #case for GPT2
                truncation=True,
                return_tensors='pt'
            ) for text in df[text_col]
        ]
    def classes(self):
        return self.labels
    def __len__(self):
        return len(self.labels)
    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])
    def get_batch_texts(self, idx):
        return self.texts[idx]
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

generated = tokenizer(Dataset, return_tensors="pt").input_ids.cuda()
model.resize_token_embeddings(len(tokenizer))
model = model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
model.train()
