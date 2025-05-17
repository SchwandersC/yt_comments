import re
import html
import emoji
import logging
from bs4 import BeautifulSoup
from nltk.corpus import words, stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger("yt_pipeline")

# Preload NLTK resources for performance
wordset = set(words.words())
stopwordset = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Cleans raw comment text by removing HTML tags, unescaping HTML entities,
    demojizing emojis, and stripping user mentions.

    Args:
        text (str): The raw text to clean.

    Returns:
        str: A cleaned version of the input text.
    """
    try:
        text = re.sub(r'<!--(.*?)-->', '', text)
        text = re.sub(r'<i>(.*?)</i>', r'\1', text)
        text = re.sub(r'<a(.*?)>(.*?)</a>', r'\2', text)
        text = html.unescape(text)
        return emoji.demojize(BeautifulSoup(text, 'html.parser').get_text(separator='\n'))
    except Exception as e:
        logger.warning(f"[clean_text] Failed to clean text: {text[:60]}... | Error: {e}")
        return ""

def clean_comments(df):
    """
    Applies text cleaning to all rows in the DataFrame's 'textDisplay' column.

    Args:
        df (pd.DataFrame): A DataFrame containing a 'textDisplay' column.

    Returns:
        pd.DataFrame: The DataFrame with cleaned text.
    """
    try:
        df['textDisplay'] = df['textDisplay'].astype(str)
        df['textDisplay'] = df['textDisplay'].str.replace(r'@\w+', '', regex=True)
        df['textDisplay'] = df['textDisplay'].apply(clean_text)
        logger.info("[clean_comments] Finished cleaning text.")
    except Exception as e:
        logger.error("[clean_comments] Error during text cleaning", exc_info=True)
    return df

def tokenize_comments(df):
    """
    Tokenizes cleaned comment text into lemmatized words, removing stopwords
    and non-English words.

    Args:
        df (pd.DataFrame): A DataFrame with a 'textDisplay' column containing cleaned text.

    Returns:
        pd.DataFrame: The DataFrame with a new 'wordtoken' column containing token lists.
    """
    def tokenizer(sentence):
        try:
            return [
                lemmatizer.lemmatize(word.lower())
                for word in sentence.split()
                if word.lower() in wordset and word.lower() not in stopwordset
            ]
        except Exception as e:
            logger.warning(f"[tokenizer] Failed to tokenize: {sentence[:60]}... | Error: {e}")
            return []

    try:
        df['wordtoken'] = df['textDisplay'].apply(tokenizer)
        empty_tokens = df[df['wordtoken'].apply(lambda x: len(x) == 0)]

        if not empty_tokens.empty:
            logger.warning(f"[tokenize_comments] {len(empty_tokens)} rows resulted in empty tokens.")
            logger.debug(empty_tokens[['textDisplay']].head(3).to_string(index=False))
        else:
            logger.info("[tokenize_comments] All rows successfully tokenized.")
    except Exception as e:
        logger.error("[tokenize_comments] Error applying tokenizer", exc_info=True)

    return df
