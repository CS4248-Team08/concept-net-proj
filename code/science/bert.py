import numpy as np
import torch
from torch.optim import Adam

from transformers import BertForSequenceClassification, DistilBertForSequenceClassification
from transformers import get_scheduler
from transformers import TrainingArguments
from datasets import load_metric

from tqdm import tqdm
from bert_dataset import get_dataloaders


train_dl, test_dl = get_dataloaders(train_test_split_frac=0.8, batch_size=256)

use_gpu = True
device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")

# define model
#model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2).to(device)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=2).to(device)
optimizer = Adam(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss().to(device)

num_epochs = 10

# training loop
model.train()
for epoch in range(num_epochs):
    for input, labels in tqdm(train_dl):
        # single input = {'input_ids':tensor, 'attention_mask': tensor} for DistillBertTokenizer
        input_id = input['input_ids'].squeeze(1).to(device)
        mask = input['attention_mask'].squeeze(1).to(device)
        labels = labels.to(device)
        # print(input_id.size(), mask.size())

        logits = model(input_id, mask).logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
# test model
metric = load_metric("accuracy")
model.eval()
for input, labels in test_dl:
    mask = input['attention_mask'].to(device)
    input_ids = input['input_ids'].squeeze(1).to(device) 
    with torch.no_grad():
        outputs = model(input_ids, mask)
        logits = outputs.logits.to("cpu")
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)

print(metric.compute())