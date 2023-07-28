import pandas as pd
import glob
files = glob.glob("./split_file_*")

frames = [pd.read_csv(f) for f in files]
pd.concat(frames).to_csv('data_cleaned.csv', index=False)