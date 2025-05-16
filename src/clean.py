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
    try:
        text = re.sub(r'<!--(.*?)-->', '', text)
        text = re.sub(r'<i>(.*?)</i>', r'\1', text)
        text = re.sub(r'<a(.*?)>(.*?)</a>', r'\2', text)
        text = html.unescape(text)
        return emoji.demojize(BeautifulSoup(text, 'html.parser').get_text(separator='\n'))
    except Exception as e:
        print(f"[clean_text] Failed to clean text: {text[:60]}... | Error: {e}")
        return ""

def clean_comments(df):
    try:
        df['textDisplay'] = df['textDisplay'].astype(str)
        df['textDisplay'] = df['textDisplay'].str.replace(r'@\w+', '', regex=True)
        df['textDisplay'] = df['textDisplay'].apply(clean_text)
    except Exception as e:
        print(f"[clean_comments] Error during text cleaning: {e}")
    return df

def tokenize_comments(df):
    def tokenizer(sentence):
        try:
            return [
                lemmatizer.lemmatize(word.lower())
                for word in sentence.split()
                if word.lower() in wordset and word.lower() not in stopwordset
            ]
        except Exception as e:
            print(f"[tokenizer] Failed to tokenize: {sentence[:60]}... | Error: {e}")
            return []

    df['wordtoken'] = df['textDisplay'].apply(tokenizer)

    # Optional: show a few empty token samples for inspection
    empty_tokens = df[df['wordtoken'].apply(lambda x: len(x) == 0)]
    if len(empty_tokens) > 0:
        print(f"[tokenize_comments] {len(empty_tokens)} rows resulted in empty tokens. Sample:")
        print(empty_tokens[['textDisplay']].head(3))

    return df
