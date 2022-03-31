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
print(type(data[id][0]))

'''
# load id -> path
sampled_problems = pickle.load(open(
            '../../data/science/paths.pkl', 'rb'), encoding='latin1')

id_to_path = dict()
print('loading problem plain texts')
for id_num in sampled_problems:
    f_short = sampled_problems[id_num]['forward']['short']
    r_short = sampled_problems[id_num]['reverse']['short']
    id_to_path[id_num+'f'] = f_short
    id_to_path[id_num+'r'] = r_short

# Update feature length to the dict
# e.g. 
embed_dim = 10
all_feature_lengths['v_graph_embed'] = embed_dim
all_feature_lengths['e_graph_embed'] = embed_dim

# Load graph embedding

v_embed_dict = {}
e_embed_dict = {}
for line in load(v_embed_file):
    word, embed = parse(line)
    v_embed_dict[word] = embed

for line in load(e_embed_file):
    word, embed = parse(line)
    e_embed_dict[word] = embed

v_id_embed_dict = {}
e_id_embed_dict = {}
v_embed_for_unknown = np.random.randn(embed_dim, dtype=np.float32)
e_embed_for_unknown = np.random.randn(embed_dim, dtype=np.float32)

for id in id_to_path.keys():
    v1, e1, v2, e2, v3, e3, v4= parse_path(id_to_path[id])

    v_id_embed_dict[id] = [v_embed_dict[v1], ..., v_embed_dict[v4]]  # use v_embed_for_unknown for missing words
    e_id_embed_dict[id] = [e_embed_dict[e1], e_embed_dict[e2], e_embed_dict[e3]]  # use e_embed_for_unknown for missing words

# save to features/ dicrector
pickle.dump(v_id_embed_dict, open("features/{xxxxx}.pkl", "rb"))
pickle.dump(e_id_embed_dict, open("features/{xxxxx}.pkl", "rb"))
'''