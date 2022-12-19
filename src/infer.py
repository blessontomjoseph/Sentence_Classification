import pandas as pd
import transformers
from transformers import AutoTokenizer,AutoModel
import pickle
import torch
import utils

# model_state = pickle.load(
    # open("../model/best_model_state.p", "rb"), map_location='cpu')
meta_data = pd.read_pickle('meta_data.p')
params=meta_data['params']
base_model=params['checkpoint']
dropout=params['dropout']
linear_input_size=768
model=utils.Model(base_model,dropout,linear_input_size)
model.load_state_dict(pickle.load(open("../model/best_model_state.p", "rb")), map_location=torch.device('cpu'))

# print(model)