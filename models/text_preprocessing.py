import re, unicodedata
from typing import List
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# uncomment if installation is needed
# nltk.download('wordnet')
# nltk.download('stopwords') 
# nltk.download('averaged_perceptron_tagger')


num_dict = {'1': 'one', '2': 'two', '3': 'three', 
            '4': 'four', '5': 'five', '6': 'six', 
            '7': 'seven', '8': 'eight', '9': 'nine', 
            '0': 'zero', '&': 'and'}

trans = str.maketrans({k: f' {v} ' \
                for k, v in num_dict.items()})  

tag_dict = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}


def remove_non_alpha(text: str) -> str:
    return re.sub('[^a-z0-9]+', ' ', text, flags=re.UNICODE)


def normalize(text: str) -> str:
    return unicodedata.normalize('NFKD', text) \
            .encode('ascii', 'ignore').decode('utf-8', 'ignore')


def remove_stopwords(words: List[str]) -> List[str]:
    return [word for word in words \
                if word not in stopwords.words('english')]


def get_wordnet_pos(word: str) -> str:
    tag = nltk.pos_tag([word])[0][1][0].upper()
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_verbs(words: List[str]) -> List[str]:
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos=get_wordnet_pos(word)) \
                for word in words]


def preprocess_sentence(text: str) -> str:
    # Character-level preprocessing
    text = text.lower()
    text = text.translate(trans) # uncomment to replace digits
    text = normalize(text) 
    text = remove_non_alpha(text)
    # Word-level preprocessing (slower)
    words = text.split()
    words = lemmatize_verbs(words)
    words = remove_stopwords(words)
    text = ' '.join(words)
    return text