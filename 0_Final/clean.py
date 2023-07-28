import pandas as pd
import sys
from tqdm.auto import tqdm

tqdm.pandas()

filename = sys.argv[1]

# read training data
df = pd.read_csv(filename)

# preprocessing text messages
import re

# function to clean text data
def clean_text(text):
    # remove URLs from text
    text = re.sub('(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', ' ', text)
    # remove mentions of other users
    text = re.sub('\B@[._a-z0-9]{3,24}', '', text)
    # replace ampersand html tag with and
    text = re.sub('\&amp;', 'and', text)
    # remove very special characters
    text = re.sub('[^A-Za-z!?\.\-]', ' ', text)
    return text

# drop rows with missing values
df = df.dropna()

# clean text data
df['text'] = df['text'].progress_apply(clean_text)
df.to_csv(filename, index=False)