import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import utils
from flask import Flask, request
import flask
import time

app = Flask(__name__)

device=torch.device('cpu')
meta_data = pd.read_pickle("meta_data.p")
params = meta_data['params']
checkpoint = params['checkpoint']
base_model = AutoModel.from_pretrained(checkpoint)
dropout = params['dropout']
linear_input_size = 768
model = utils.Model(base_model, dropout, linear_input_size)
model.to(device)
model.load_state_dict(torch.load(open("../model/model_params", "rb"), map_location=device))
tokenizer=AutoTokenizer.from_pretrained(checkpoint)


def mapper(val):
        mapper_ = {0: 'neutral',
              1: 'entailment', 
              2: 'contradiction'}
        return mapper_[val]


def compute(input):
    model.eval()
    sent1=input[0]
    sent2=input[1]
    batch = tokenizer(text=sent1, text_pair=sent2, truncation=True,padding='max_length', max_length=200, return_tensors='pt')
    input={'x': {k: v.squeeze(dim=0).to(device) for k, v in batch.items()}}
    out=model(input['x'])
    out=torch.nn.Softmax(out,dim=0)
    index=torch.argmax(out,dim=0) 
    result=map(mapper,index)
    return result
    
@app.route("/predicts",methods=["GET"])
def predict():
    key=request.args.get('key')
    start=time.time()
    ans=compute(key)
    response={}
    response['response']={'key':key,
                          'time':time.time()-start,
                          'ans':ans              
                          
                          }
    
    return flask.jsonify(response)

if __name__=="__main__":
    app.run(host="0.0.0.0")  
