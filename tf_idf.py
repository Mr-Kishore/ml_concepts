import math
corpus = [
    'hello everyone I welcome you all here',
    'it is a wonderful meetup for everyone',
    'today is a sunny day',
    'we are going to play basketball'
]

#Build the word set
word_set = set()
for doc in corpus:
    words = doc.split()
    word_set.update(words)

print(len(word_set))
print(word_set)

#Term Frequency (TF)
n_docs = len(corpus)
n_word_set = len(word_set)

tf = []

for doc in corpus:
    words = doc.split()
    doc_length = len(words)
    tf_doc = {}
    
    for word in words:
        if word not in tf_doc:
            tf_doc[word] = 0
        tf_doc[word] += 1 / doc_length 
        
    tf.append(tf_doc)

print("Term Frequency (TF):")
for i, tf_doc in enumerate(tf):
    print(f"Document {i+1}: {tf_doc}")


print("\nIDF of:")
idf = {}

for word in word_set:
    k = sum(1 for doc in corpus if word in doc.split())
    idf[word] = 0 if k == 0 else math.log10(n_docs / k)  # Avoid division by zero
    print(f'{word:>15}: {idf[word]:>10}')

