from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/stopwords')
except LookupError:
    nltk.download('stopwords')

import docx
from simplify_docx import simplify

def doc_prep(file: str):
    my_doc = docx.Document(file)
    my_doc_as_json = simplify(my_doc)
    breakpoint()


def filter_keywords(file: str) -> str:

    fp = open(file, encoding='UTF-8')
    raw_text = fp.read()
    # Tokenize
    sentences = sent_tokenize(raw_text)
    print(f"N Sen = {len(sentences)}\n")

    stop_words = set(stopwords.words('english'))

    filtered_sentences = []

    # Extract Keywords per tokens
    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_sentence = [word for word in words if word.lower() not in stop_words]
        filtered_sentences.append(filtered_sentence)

    #up until now filtered_sentences is a list of keywords

    #join the keywords to return a single string object
    preprocessed_text = [' '.join(sentence) for sentence in filtered_sentences]

    return sentences,preprocessed_text

sen,res = filter_keywords("Human_Nature.txt")


breakpoint()
