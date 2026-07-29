"""
===================================================================
Name: BloomTaxonomy slim-artifact builder
Author: MLG Castanares
Description: Converts the 147MB training pickle into a small inference-only
             artifact that fits in a public GitHub repo.
Date: 2026-07-29
====================================================================
Why this exists
---------------
XGBLau_TFIDF.pkl is a *training* record: it carries X_train (110MB) and
X_test (34MB) as dense float64 arrays. Inference needs none of that, only:

    model       the fitted XGBClassifier                      (~1MB)
    vectorizer  the TF-IDF fitted on df_train                 (~1MB)
    scaler      the StandardScaler fitted on X_train           (~0.1MB)

Fitting the vectorizer once here also removes it from the hot path: the
original code refit TF-IDF over all 4321 training docs (POS-tagging and
stemming each one) on *every* predict_proba call, which LIME makes hundreds
of times per explanation.

Usage:  python build_slim_model.py
Output: model_slim.pkl  (committed to the repo; the 147MB pickle is not)
====================================================================
"""
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from utils import enhanced_preprocess

FULL_MODEL_PATH = './XGBLau_TFIDF.pkl'
SLIM_MODEL_PATH = './model_slim.pkl'
CLASS_NAMES = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
FEAT = 'text'


def build(full_path=FULL_MODEL_PATH, slim_path=SLIM_MODEL_PATH):
    print(f'Loading {full_path} ...')
    with open(full_path, 'rb') as f:
        out = pickle.load(f)

    # Refit the vectorizer exactly as proc_TFIDF does, but keep the fitted
    # object instead of throwing it away.
    print('Fitting TF-IDF vectorizer on df_train ...')
    vectorizer = TfidfVectorizer(preprocessor=enhanced_preprocess)
    X_train = vectorizer.fit_transform(out['df_train'][FEAT])
    X_train = X_train.toarray()

    # Sanity check: the refit must reproduce the stored training matrix,
    # otherwise the scaler and the model would see different features.
    stored = np.asarray(out['X_train'])
    if X_train.shape != stored.shape:
        raise RuntimeError(f'shape mismatch: refit {X_train.shape} vs stored {stored.shape}')
    max_diff = float(np.abs(X_train - stored).max())
    print(f'  refit vs stored X_train: shape {X_train.shape}, max abs diff {max_diff:.2e}')
    if max_diff > 1e-8:
        raise RuntimeError('refit TF-IDF does not reproduce the stored X_train')

    print('Fitting StandardScaler ...')
    scaler = StandardScaler()
    scaler.fit(stored)

    # Store the booster in XGBoost's own portable format rather than as a
    # pickled XGBClassifier. Pickled boosters warn (and may break) when loaded
    # by a different XGBoost version, which is exactly what happens on a cloud
    # deploy that installs the current release.
    print('Serialising booster to portable UBJ ...')
    model_raw = bytes(out['model'].get_booster().save_raw(raw_format='ubj'))

    slim = {
        'name': out.get('name'),
        'model_raw': model_raw,
        'vectorizer': vectorizer,
        'scaler': scaler,
        'class_names': CLASS_NAMES,
        'report': out.get('report'),
        'sklearn_version': __import__('sklearn').__version__,
    }

    with open(slim_path, 'wb') as f:
        pickle.dump(slim, f, protocol=pickle.HIGHEST_PROTOCOL)

    import os
    size_mb = os.path.getsize(slim_path) / 1e6
    full_mb = os.path.getsize(full_path) / 1e6
    print(f'\nWrote {slim_path}: {size_mb:.1f}MB (from {full_mb:.0f}MB)')
    if size_mb > 90:
        print('WARNING: still close to GitHub\'s 100MB per-file limit.')


if __name__ == '__main__':
    build()
