import torch
from dataset import Dataset
from learn import train, test


# torch.autograd.set_detect_anomaly(True)
use_gpu = True
device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

features = ['v_enc_dim300', 'v_freq_freq', 'v_deg', 'v_sense', 'e_vertexsim',
            'e_dir', 'e_rel', 'e_weightsource', 'e_srank_rel', 'e_trank_rel', 'e_sense']
split_frac = 0.8
dataset = Dataset(features, split_frac, device)

feature_enc_len = 20
feature_enc_type = 'proj+mean'  # 'proj+mean' OR 'concat+proj'
path_enc_type = "LSTM"  # 'RNN' OR 'LSTM' OR 'Attention'

num_epoch = 400
N = 1024  # batch size
num_iter = num_epoch * dataset.train_size//N
print(f"Config: feature_enc_len:{feature_enc_len}, path_enc_type:{path_enc_type}, feature_enc_type:{feature_enc_type}, N:{N}, n_epoch:{num_epoch}")

encoder, predictor, loss = train(dataset, feature_enc_len, num_iter, N, device, path_enc_type, feature_enc_type)

config = [feature_enc_len, feature_enc_type, path_enc_type, N, num_epoch]
test(dataset, encoder, predictor, loss, config)


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