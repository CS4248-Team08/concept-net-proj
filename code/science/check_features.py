import pickle

all_feature_lengths = {'v_enc_onehot': 100,
                       'v_enc_embedding': 300,
                       'v_enc_dim300': 300,
                       'v_enc_dim2': 2,
                       'v_enc_dim10': 10,
                       'v_enc_dim50': 50,
                       'v_enc_dim100': 100,
                       'v_freq_freq': 1,
                       'v_freq_rank': 1,
                       'v_deg': 1,
                       'v_sense': 1,
                       'e_vertexsim': 1,
                       'e_dir': 3,
                       'e_rel': 46,
                       'e_weight': 1,
                       'e_source': 6,
                       'e_weightsource': 6,
                       'e_srank_abs': 1,
                       'e_srank_rel': 1,
                       'e_trank_abs': 1,
                       'e_trank_rel': 1,
                       'e_sense': 1}

fea_name = "e_dir"
data = pickle.load(open(f"features/{fea_name}.pkl", "rb"), encoding="latin1")
id = "114409r"

print(data[id])  # 3 edges, each contains the feature