"""
===================================================================
Name: BloomTaxonomy Demo App (Streamlit)
Author: MLG Castanares
Description: Takes a text input, predicts its Bloom level and shows
             the LIME explanation for the predicted class.
Date: 2026-07-29
====================================================================
Public repo
====================================================================
Run with:  streamlit run streamlit_app.py

Model artifact
--------------
Prefers model_slim.pkl (1MB, inference-only: model + vectorizer + scaler),
which is what gets committed and deployed. Falls back to the full 147MB
training pickle XGBLau_TFIDF.pkl when present locally.
Rebuild the slim artifact with:  python build_slim_model.py
====================================================================
"""
import os
import pickle
import threading
import hmac

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from lime.lime_text import LimeTextExplainer

from utils import clean_text, proc_TFIDF


# Resolve data files against this file's directory, not the working directory.
# Streamlit Community Cloud runs the app with cwd set to the REPO ROOT
# (/mount/src/github_apps), so './XGBLau_slim.pkl' would look in the wrong
# folder for an app that lives in a subdirectory.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SLIM_MODEL_PATH = os.path.join(APP_DIR, 'XGBLau_slim.pkl')
CLASS_NAMES = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']

# NLTK corpora needed by utils.enhanced_preprocess (word_tokenize + pos_tag).
# On Streamlit Community Cloud these are installed from nltk.txt; this is the
# fallback for local runs and any environment that skips that file.
NLTK_RESOURCES = [
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab'),
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
]


@st.cache_resource(show_spinner='Downloading language data...')
def ensure_nltk_data():
    import nltk
    for path, package in NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)
    return True


# HELPER FUNCTIONS
class PredictProbaWrapperSlim:
    '''Inference-only predictor built on the slim artifact.

    The vectorizer and scaler are already fitted, so a call is just
    clean -> transform -> scale -> predict_proba.
    '''
    def __init__(self, mod_filepath):
        import xgboost as xgb

        with open(mod_filepath, 'rb') as f:
            out = pickle.load(f)

        self.out = out
        # The booster ships as portable UBJ bytes so it loads cleanly on
        # whatever XGBoost version the deploy environment installs.
        self.model = xgb.XGBClassifier()
        self.model.load_model(bytearray(out['model_raw']))
        self.vectorizer = out['vectorizer']
        self.scaler = out['scaler']
        self.class_names = out.get('class_names', CLASS_NAMES)

    def __call__(self, text_list):
        cleaned = [clean_text(t) for t in text_list]
        X_test = self.vectorizer.transform(cleaned)
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()

        X_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_scaled)


class PredictProbaWrapperML:
    '''This wrapper:
    - loads the model pickle file
    - prepares the proper scaler and vectorizer based on training data
    - returns the predict_proba function used by LIME for explanations

    Kept for the full training pickle. Note it refits the TF-IDF vectorizer on
    every call, so it is far slower than PredictProbaWrapperSlim.
    '''
    def __init__(self, mod_filepath, fn_preprocess, feat=['text']):
        from sklearn.preprocessing import StandardScaler
        with open(mod_filepath, 'rb') as f:
            out = pickle.load(f)

        self.out = out
        self.model = out['model']
        self.scaler = StandardScaler()
        self.scaler.fit(out['X_train'])
        self.fn_preprocess = fn_preprocess
        self.feat = feat
        self.class_names = CLASS_NAMES

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

    @property
    def class_names(self):
        return getattr(self.predict_fn, 'class_names', CLASS_NAMES)


@st.cache_resource(show_spinner='Loading model...')
def load_predictor():
    '''Loads the model once per process and wraps it with a lock so concurrent
    Streamlit reruns cannot call predict_proba at the same time.'''
    if os.path.exists(SLIM_MODEL_PATH):
        predict_proba_fn = PredictProbaWrapperSlim(SLIM_MODEL_PATH)
    else:
        raise FileNotFoundError(
            f'No model artifact found at {SLIM_MODEL_PATH}. '
            f'Build it with: python build_slim_model.py')

    return ThreadSafePredictProbaWrapper(predict_proba_fn, threading.Lock())


@st.cache_resource
def get_explainer():
    return LimeTextExplainer(class_names=CLASS_NAMES)


def explain(text, predict_proba_fn, num_features, num_samples):
    '''Single inference call + LIME explanation for the predicted class.'''
    probs = predict_proba_fn([text])[0]
    pred_class_idx = int(np.argmax(probs))

    exp = get_explainer().explain_instance(
        text,
        predict_proba_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=[pred_class_idx],
    )
    return probs, pred_class_idx, exp

def get_setting(key, default=None):
    """Read a config value from st.secrets (Streamlit Cloud) or the environment."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

def check_password():
    """Return True if the user is authorized.

    The password is read from st.secrets["APP_PASSWORD"] (or the APP_PASSWORD
    env var). If none is configured, the gate is open — so local dev works
    without secrets while a deployed instance can require one.
    """
    password = get_setting("APP_PASSWORD")
    if not password:
        return True
    if st.session_state.get("password_correct"):
        return True

    def _verify():
        entered = st.session_state.get("password", "")
        if hmac.compare_digest(str(entered), str(password)):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't keep the raw password around
            # The login *is* the session — begin with an empty store.
        else:
            st.session_state["password_correct"] = False

    st.title("ML-based Bloom Taxonomy Classifier")
    st.caption("This app predicts the Bloom level of a learning objective or question.")
    st.caption("Designed by M. Castanares")
    st.caption("Please enter the password to access the app.")
    st.text_input("Password", type="password", on_change=_verify, key="password")
    
    if st.session_state.get("password_correct") is False:
        st.error("😕 Incorrect password.")
    
    return False

# APP
st.set_page_config(page_title="ML-based Bloom Taxonomy Classifier")

if not check_password():
    st.stop()

num_features= 6
num_samples= 100

st.set_page_config(page_title='Bloom Taxonomy Classifier',
                   layout='wide')
st.title('ML-based Bloom Taxonomy Classifier')
st.caption('**About this app**. This app predicts the Bloom level of a learning objective or question '
           'and provides a LIME explanation of the prediction.')
st.caption('**Instruction**. Enter a learning objective or question to get its Bloom level '
           '(i.e., remember, understand, apply, analyze, evaluate, create). '
           'A sample result is shown below. '
           'The model predicts that the statement is at "create" Bloom level (0.59 probability), '
           'attributed to the words "design", "optimized", and "establish".')
st.image('https://github.com/MichaelCastanares/Github_apps/blob/5daf6e53712b9b924ca54331b26795b802142b2f/app_BloomTaxonomy/example.png?raw=true', use_column_width=True)

text = st.text_area('Text to classify', 
                    value='Participants will design an optimized project timeline using Gantt charts and establish key milestones for deliverables',
                    height=100, placeholder='e.g. Compare the two theories...')
run = st.button('Classify', type='primary')
if run:
    if not text.strip():
        st.warning('Please enter some text.')
        st.stop()

    ensure_nltk_data()

    try:
        predict_proba_fn = load_predictor()
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()

    with st.spinner('Running model and LIME...'):
        probs, pred_class_idx, exp = explain(text, predict_proba_fn,
                                             num_features, num_samples)

    st.subheader('LIME explanation')
    st.caption(f'Word contributions towards **{CLASS_NAMES[pred_class_idx]}** prediction.')

    components.html(exp.as_html(labels=[pred_class_idx]), height=450,
                    scrolling=False)

st.caption('**Final note**. The pipeline demonstrates the use of ML models for scalable assessment '
           'of learning objectives and questions against Pedagogical Frameworks. '
           'The LIME tool provides interpretability of the predictions. '
           'This tool should be used with instructor oversight.')