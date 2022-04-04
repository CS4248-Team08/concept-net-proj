from __future__ import division

import numpy as np
import torch
import sys
import os
from torch import nn, optim
from torch.autograd import Variable
from model import ChainEncoder, Predictor, JointModel
from dataset import Dataset
from multiprocessing import Pool
import time
from datetime import datetime
import pickle
from sklearn.metrics import f1_score, recall_score, precision_score


def train(dataset, fea_len, num_iter=4000, N=1000, learning_rate=0.001, device='cuda', path_enc_type="LSTM",
          feature_enc_type='proj+mean', out_file='train.log'):
    if isinstance(out_file, str):
        out_file = open(out_file, 'w')
    out_file.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    print('defining architecture')
    encoder = ChainEncoder(dataset.get_v_fea_len(),
                           dataset.get_e_fea_len(),
                           out_length=fea_len, pooling='mean',
                           path_encode_type=path_enc_type, feature_encode_type=feature_enc_type)
    predictor = Predictor(fea_len)
    # model = JointModel(dataset.get_v_fea_len(), dataset.get_e_fea_len(), fea_len, 'last')
    loss = nn.NLLLoss()

    encoder.to(device=device)
    predictor.to(device=device)
    # model.to(device=device)
    loss.to(device=device)

    optimizer = optim.Adam(list(encoder.parameters()) +
                           list(predictor.parameters()), lr=learning_rate)
    # optimizer = optim.Adam(model.parameters())
    iters_per_epoch = dataset.train_size // N
    print('Start training')
    start = time.time()
    encoder.train()
    predictor.train()
    epoch_loss = 0
    for train_iter in range(num_iter):
        chains_A, chains_B, y = dataset.get_train_pairs(N)
        val_A, val_B, val_y = dataset.get_test_pairs(randomize_dir=True, return_id=False)
        val_y_cpu = val_y.to('cpu').numpy()

        output_A = encoder(chains_A)
        output_B = encoder(chains_B)
        logSoftmax_output = predictor(output_A, output_B)
        optimizer.zero_grad()
        # logSoftmax_output = model(chains_A, chains_B)
        loss_val = loss(logSoftmax_output, y)
        loss_val.backward()
        optimizer.step()
        epoch_loss += loss_val.item()

        if train_iter > 0 and (train_iter + 1) % iters_per_epoch == 0:
            encoder.eval()
            predictor.eval()
            with torch.no_grad():
                val_A = encoder(val_A)
                val_B = encoder(val_B)
                val_output = predictor(val_A, val_B)
                val_loss = loss(val_output, val_y)
                y = y.to('cpu').numpy()
                train_pred = logSoftmax_output.to('cpu').numpy().argmax(axis=1)
                val_pred = val_output.to('cpu').numpy().argmax(axis=1)
                train_acc = (train_pred == y).sum() / len(y)
                val_acc = (val_pred == val_y_cpu).sum() / len(val_y)

                print(
                    f"Progress: {100 * (train_iter+1) / num_iter:.2f}%, loss: {epoch_loss}, acc: {train_acc}, time spent: {(time.time() - start) / 60:.2f} minutes")

                out_file.write(
                    f"{(train_iter+1) // iters_per_epoch},{epoch_loss / (iters_per_epoch * N)},{train_acc},{val_loss / len(val_y)},{val_acc}\n")
                epoch_loss = 0
                torch.save(encoder.state_dict(),
                           f'ckpt/{train_iter}_encoder.model')
                torch.save(predictor.state_dict(),
                           f'ckpt/{train_iter}_predictor.model')

            encoder.train()
            predictor.eval()

    print(f'Finish training, time spent: {(time.time() - start) / 60:.2f} minutes')
    out_file.close()
    return encoder, predictor, loss


def test(dataset, encoder, predictor, loss, config=None, out_file='test.log'):
    if isinstance(out_file, str):
        out_file = open(out_file, 'a')

    if config:
        feature_enc_len, feature_enc_type, path_enc_type, N, num_epoch = config

    print("Start testing")
    chains_A, chains_B, y = dataset.get_test_pairs(randomize_dir=True, return_id=False)

    encoder.eval()
    with torch.no_grad():
        output_test_A = encoder(chains_A)
        output_test_B = encoder(chains_B)
        logSoftmax_output = predictor(
            output_test_A, output_test_B).to(device='cpu').numpy()

        pred = logSoftmax_output.argmax(axis=1)
        y = y.to(device='cpu').numpy()

        cur_acc = (pred == y).sum() / len(y)
        recall = recall_score(y, pred)
        precision = precision_score(y, pred)
        f1 = f1_score(y, pred)

        print(f'Test accuracy: {cur_acc}, recall: {recall}, precision: {precision}, f1: {f1}')
        out_file.write("Test time: {}\n".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        out_file.write(
            f"Test config: feature_enc_len:{feature_enc_len}, feature_enc_type:{feature_enc_type}, path_enc_type:{path_enc_type}, N:{N}, epoch:{num_epoch}\n")
        out_file.write(f'Test accuracy: {cur_acc}, recall: {recall}, precision: {precision}, f1: {f1}\n\n')

    out_file.close()
