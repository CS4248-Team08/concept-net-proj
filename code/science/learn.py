
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

def train(dataset, fea_len, num_iter=4000, N=1000, path_enc_type="LSTM", out_file='train.log'):
    if isinstance(out_file, str):
        out_file = open(out_file, 'w')
    out_file.write("n_iter,loss\n")

    print('defining architecture')
    encoder = ChainEncoder(dataset.get_v_fea_len(),
                           dataset.get_e_fea_len(), 
                           out_length=fea_len, pooling='mean', path_encoder_type=path_enc_type)
    predictor = Predictor(fea_len)
    # model = JointModel(dataset.get_v_fea_len(), dataset.get_e_fea_len(), fea_len, 'last')
    loss = nn.NLLLoss()

    encoder.to(device=device)
    predictor.to(device=device)
    # model.to(device=device)
    loss.to(device=device)

    optimizer = optim.Adam(list(encoder.parameters()) +
                           list(predictor.parameters()))
    # optimizer = optim.Adam(model.parameters())

    print('Start training')
    start = time.time()
    encoder.train()
    for train_iter in range(num_iter):
        chains_A, chains_B, y = dataset.get_train_pairs(N)
        output_A = encoder(chains_A)
        output_B = encoder(chains_B)
        logSoftmax_output = predictor(output_A, output_B)
        optimizer.zero_grad()
        # logSoftmax_output = model(chains_A, chains_B)
        loss_val = loss(logSoftmax_output, y)
        loss_val.backward()
        optimizer.step()

        if train_iter % 100 == 0:
            print(
                f"Progress: {100*train_iter/num_iter:.2f}%, loss: {loss_val.item()}, time spent: {(time.time() - start)/60:.2f} minutes")

            out_file.write(f"{train_iter}, {loss_val.item()}\n")
            torch.save(encoder.state_dict(),
                       f'ckpt/{train_iter}_encoder.model')
            torch.save(predictor.state_dict(),
                       f'ckpt/{train_iter}_predictor.model')

    print(f'Finish training, time spent: {(time.time()-start)/60:.2f} minutes')
    out_file.close()
    return encoder, predictor, loss


def test(dataset, encoder, predictor, loss, out_file='test.log'):
    if isinstance(out_file, str):
        out_file = open(out_file, 'a')

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
        out_file.write(f"Test config: feature_enc_len:{feature_enc_len}, path_enc_type:{path_enc_type}, N:{N}, epoch:{num_epoch}\n")
        out_file.write(f'Test accuracy: {cur_acc}, recall: {recall}, precision: {precision}, f1: {f1}\n\n')

    out_file.close()


#torch.autograd.set_detect_anomaly(True)
use_gpu = True
device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

features = ['v_enc_dim300', 'v_freq_freq', 'v_deg', 'v_sense', 'e_vertexsim',
            'e_dir', 'e_rel', 'e_weightsource', 'e_srank_rel', 'e_trank_rel', 'e_sense']
split_frac = 0.8
dataset = Dataset(features, split_frac, device)

feature_enc_len = 20
num_epoch = 300
N = 1024  # batch size
num_iter = num_epoch * dataset.train_size//N
path_enc_type = "LSTM"
print(f"Config: feature_enc_len:{feature_enc_len}, path_enc_type:{path_enc_type}, N:{N}, n_epoch:{num_epoch}")

encoder, predictor, loss = train(dataset, feature_enc_len, num_iter, N, path_enc_type)
test(dataset, encoder, predictor, loss)


################ Try plotting the attention heatmap ################################

# chains_A, chains_B, y = dataset.get_test_pairs(randomize_dir=True, return_id=False)
# embed_A, att_A = encoder.forward(chains_A, return_attention=True)
# embed_B, att_B = encoder.forward(chains_B, return_attention=True)

# sample_i = 0
# att_A = [torch.squeeze(att[sample_i]).cpu().detach().numpy() for att in att_A]  # remove batch dim
# att_B = [torch.squeeze(att[sample_i]).cpu().detach().numpy() for att in att_B]
# with open("att_A.pkl", "wb") as f:
#     pickle.dump(att_A, f)
# with open("att_B.pkl", "wb") as f:
#     pickle.dump(att_B, f)