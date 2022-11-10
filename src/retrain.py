import pickle
import train
import config
import numpy as np
import pandas as pd

if __name__=="__main__":
    meta_data=pd.read_pickle('meta_data.p')
    best_params=meta_data['params']
    best_model_state=None
    best_score=0
    info={}
    for fold in range(config.nsplits):
        score,model_state = train.run_training(fold, best_params)
        if score>best_score:
            best_score=score
            best_model_state=model_state
            info['model_score']=score
            info['on_fold']=fold
            
pickle.dump(best_model_state,open("model/best_model_state.p","wb"))
pickle.dump(info,open("model/info.p","wb"))


