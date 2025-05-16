import re
import html
import emoji
from bs4 import BeautifulSoup
from nltk.corpus import words, stopwords
from nltk.stem import WordNetLemmatizer

wordset = set(words.words())
stopwordset = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'<!--(.*?)-->', '', text)
    text = re.sub(r'<i>(.*?)</i>', r'\1', text)
    text = re.sub(r'<a(.*?)>(.*?)</a>', r'\2', text)
    text = html.unescape(text)
    return emoji.demojize(BeautifulSoup(text, 'html.parser').get_text(separator='\n'))

def clean_comments(df):
    df['textDisplay'] = df['textDisplay'].str.replace(r'@\w+', '', regex=True)
    df['textDisplay'] = df['textDisplay'].apply(clean_text)
    return df

def tokenize_comments(df):
    def tokenizer(sentence):
        return [lemmatizer.lemmatize(word.lower()) for word in sentence.split()
                if word.lower() in wordset and word.lower() not in stopwordset]
    df['wordtoken'] = df['textDisplay'].apply(tokenizer)
    return df
