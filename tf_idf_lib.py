import pandas as pd
import numpy as np

corpus = ['hello everyone I welcome you all here',
          'it is a wonderful meetup for everyone']

word_set = set()

for doc in corpus:
    word = doc.split()
    word_set = word_set.union(set(word))

print(len(word_set))
print(word_set)

n_docs = len(corpus)
n_word_set = len(word_set)

# Create the Term Frequency DataFrame with columns as a list, not a set
df_tf = pd.DataFrame(np.zeros((n_docs, n_word_set)), columns=list(word_set))


for i in range(n_docs):
    words = corpus[i].split(' ')
    for w in words:
        df_tf.loc[i, w] = df_tf.loc[i, w] + (1 / len(words))


print(df_tf)

print("IDF of: ")

idf = {}

for w in word_set:
    k = 0    # number of documents in the corpus that contain this word
    
    for i in range(n_docs):
        if w in corpus[i].split():
            k += 1
            
    idf[w] =  np.log10(n_docs / k)
    
    print(f'{w:>15}: {idf[w]:>10}' )