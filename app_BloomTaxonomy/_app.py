"""
===================================================================
Name: BloomTaxonomy Demo App
Author: MLG Castanares
Description: Runs trained classifier models to assess Bloom Levels of Text
Date: 2026-07-29
====================================================================
Public repo
====================================================================
"""
import sys
import pandas as pd
import numpy as np
from lime.lime_text import LimeTextExplainer

from utils import clean_text, enhanced_preprocess, proc_TFIDF


#HELPER FUNCTIONS
class PredictProbaWrapperML:
    '''This wrapper:
    - loads the model pickle file
    - prepares the proper scaler and vectorizer based on training data
    - returns the predict_proba function used by LIME for explanations
    '''
    def __init__(self, mod_filepath, fn_preprocess, feat=['text']):
        from sklearn.preprocessing import StandardScaler
        import pickle
        with open(mod_filepath, 'rb') as f:
            out = pickle.load(f)
        
        self.out = out
        self.model = out['model']
        self.scaler = StandardScaler()
        self.scaler.fit(out['X_train'])
        self.fn_preprocess = fn_preprocess
        self.feat = feat
    def __call__(self, text_list):

        df_test = pd.DataFrame({'text': text_list})
        df_test['text'] = df_test['text'].apply(clean_text)
        _, X_test = self.fn_preprocess(self.out['df_train'], df_test, self.feat)
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()
        
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_scaled)


class ThreadSafePredictProbaWrapper:
    '''Wrapper to make predict_proba calls thread-safe using a lock'''
    def __init__(self, predict_fn, lock):
        self.predict_fn = predict_fn
        self.lock = lock
    
    def __call__(self, text_list):
        with self.lock:
            return self.predict_fn(text_list)




# read text
text = ["Discuss the following"]

# load models
MOD_FILE_PATH =  './XGBLau_TFIDF.pkl'
PATH_LIME_NB = f'./LIME_tmp.html'

class_names = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
explainer = LimeTextExplainer(class_names=class_names)



if MOD_FILE_PATH.endswith('.pkl'):
    predict_proba_fn = PredictProbaWrapperML(mod_filepath=MOD_FILE_PATH,
                                       fn_preprocess=proc_TFIDF)
# elif MOD_FILE_PATH.endswith('.pt'):
#     predict_proba_fn = PredictProbaWrapperTransformers(mod_filepath=MOD_FILE_PATH)
else:
    raise ValueError(f"Unsupported model type, {MOD_FILE_PATH}")


# run lime
explainer_obj=explainer,
target_class = 0

# Single inference call
probs = predict_proba_fn([text])[0]
pred_class_idx = np.argmax(probs)

print(probs)
print(pred_class_idx)

# OPTIMIZED: Reduce num_samples from default 5000 to 1000 (2-3x speedup)
exp = explainer_obj[0].explain_instance(
    text[0], 
    predict_proba_fn,
    num_features=6,
    num_samples=100,  # ← Reduced from default
    labels=[pred_class_idx]
)
exp.save_to_file(PATH_LIME_NB)



#