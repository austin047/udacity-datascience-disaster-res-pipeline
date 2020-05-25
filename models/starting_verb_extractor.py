from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import nltk


# Cuustom Estimator 'StartingVerbExtractor' that extracts the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Description: Custom Estimator 'StartingVerbExtractor' that extracts the starting verb of a sentence
    
    Baseclasses:
        BaseEstimator
        TransformerMixin
    '''
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        

    def starting_verb(self, text):
        # Extract sentences from text
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            # Extract words into seperate tags
            pos_tags = nltk.pos_tag(self.tokenizer(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged) 
    