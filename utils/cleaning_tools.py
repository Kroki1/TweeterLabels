import re

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
# nltk.download()
stemmer = PorterStemmer()
repl_dict = {"%20": " ",
             "&gt;": "",
            # "??": "?",
             r'\r': ' ',
             r'\x89': "",
             r'\n': ' ',
             r'\x9d': "",
             "_": "",
             }


def remove_http(text):
    http = "https?://\S+|www\.\S+" # matching strings beginning with http (but not just "http")
    pattern = r"({})".format(http) # creating pattern
    return re.sub(pattern, "", text)


def remove_stopwords(text):
    # nltk.download('stopwords')
    stops = stopwords.words("english")  # stopwords
    addstops = ["among", "onto", "shall", "thrice", "thus", "twice", "unto", "us", "would"]  # additional stopwords
    allstops = stops + addstops
    return " ".join([word for word in text.split() if word not in allstops])


def text_stemmer(text):
    text_stem = " ".join([stemmer.stem(word) for word in text.split()])
    return text_stem


def lemmatize_text(text):
    this = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
    return this

