import pandas as pd

# Read in the data
df = pd.read_csv('data.csv')

S=50000 # length of each split
N = int(len(df)/S)

# Split the data into N parts
frames = [ df.iloc[i*S:(i+1)*S].copy() for i in range(N) ]

sum = 0
for i in range(N):
    frames[i].to_csv('split_file_' + str(i) + '.csv', index=False)
    print(len(frames[i]))
    sum += len(frames[i])

print(sum)
