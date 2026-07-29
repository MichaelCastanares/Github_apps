import re
import nltk
from nltk.stem import PorterStemmer

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation but keep words/numbers
    text = re.sub(r"[^\w\s]", "", text)
    # 3. Normalize whitespace (remove multiple spaces, tabs, newlines)
    text = re.sub(r"\s+", " ", text)
    # 4. Strip leading/trailing spaces
    text = text.strip()

    return text

def enhanced_preprocess(text):
    ''' The function takes a text input and applies tokenization,
    POS tagging, and stemming and applies Mohammed's "Impact Factor"
    by weighting verbs more heavily than nouns and adjectives.

    w1 = 5(verbs), w2 = 3(nouns/adjectives) w3 = 1(others)

    parameters
    ----------
    text: str - The input text to be processed.

    returns
    -------
    str - The processed text with enhanced weighting for verbs and nouns/adjectives.
    '''



    tokens = nltk.word_tokenize(text.lower())
    pos_tags = nltk.pos_tag(tokens)
    stemmer = PorterStemmer()

    enhanced_tokens = []
    for word, tag in pos_tags:
        stemmed = stemmer.stem(word)
        # Apply Mohammed's "Impact Factor": repeat verbs to increase their TF
        if tag.startswith('VB'):  # Verbs
            enhanced_tokens.extend([stemmed] * 5) # Triple the weight for verbs
        elif tag.startswith('NN') or tag.startswith('JJ'): # Nouns/Adjectives
            enhanced_tokens.extend([stemmed] * 3) # Double the weight for nouns/adjectives
        else:
            enhanced_tokens.append(stemmed)

    return " ".join(enhanced_tokens)

def proc_TFIDF(df_train, df_test, feat_name):
    from sklearn.feature_extraction.text import TfidfVectorizer

    #takes only the first feat == 'text'
    if isinstance(feat_name, list):
        feat_name = feat_name[0]

    vectorizer = TfidfVectorizer(preprocessor=enhanced_preprocess)
    X_train = vectorizer.fit_transform(df_train[feat_name])
    X_test = vectorizer.transform(df_test[feat_name])
    return X_train, X_test
