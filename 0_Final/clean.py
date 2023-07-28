import pandas as pd
import sys
from tqdm.auto import tqdm

tqdm.pandas()

filename = sys.argv[1]

# read training data
df = pd.read_csv(filename)

# preprocessing text messages
import re
import nltk
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer

# download stopwords and wordnet
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

# create object of WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# this allows very
stopwords = set(stopwords.words('english'))
stopwords.remove('very')
print(stopwords)

correct_words = [str.lower(w) for w in words.words()]

# BEGIN SOURCE http://norvig.com/spell-correct.html
from collections import Counter

words = Counter(correct_words)


def P(word, N=sum(words.values())):
    """Probability of `word`."""
    return words[word] / N


def correction(word):
    """Most probable spelling correction for word."""
    return max(candidates(word), key=P)


def candidates(word):
    """Generate possible spelling corrections for word."""
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def known(words):
    """The subset of `words` that appear in the dictionary of WORDS."""
    return set(w for w in words if w in words)


def edits1(word):
    """All edits that are one edit away from `word`."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """All edits that are two edits away from `word`."""
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


# END SOURCE

# function to clean text data
def clean_text(text):
    # remove mentions of other users
    text = re.sub('\B@[._a-zA-Z0-9]{3,24}', '', text)

    # rewrite words in all caps to "very" followed by word
    # text = re.sub('([A-Z]+)', lambda x: 'very ' + x.group(0).lower(), text)

    # make words lowercase, because Go and go will be considered as two words
    text = text.lower()

    # remove multiple dots
    text = re.sub(r'(\.)\1{2,}', '\1', text)

    # remove URLs from text (prefer safely!)
    text = re.sub('https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', ' ',
                  text)

    # replace ampersand html tag with &
    text = re.sub('\&amp;', 'and', text)

    # remove everything but letters
    text = re.sub('[^a-z]', ' ', text)

    # split the sentences into words
    _words = text.split()

    for i in range(len(_words)):
        # remove words with length 1
        if len(_words[i]) == 1:
            _words[i] = ''

        # remove repetition of letters
        _tmp = re.subn(r'([a-z])\1{3,}', r'\1', _words[i])

        # prepend very if there were repeating letters
        # if _tmp[1] > 0:
        #     words[i] = "very " + _tmp[0]

    # remove stopwords like to, and, or etc.
    # _words = [word for word in _words if word not in stopwords]
    _words = [correction(word) for word in _words]

    # remove words if they are unknown
    # _words = ['' if word not in words else word for word in _words]

    # lemmatize each word
    # words = [wordnet_lemmatizer.lemmatize(word) for word in words]

    # join words to make sentence
    text = ' '.join(_words)
    # remove multiple spaces
    text = re.sub('\s+', ' ', text)
    text = text.strip()

    return text

# drop rows with missing values
df = df.dropna()

# clean text data
df['text'] = df['text'].progress_apply(clean_text)
df.to_csv(filename, index=False)