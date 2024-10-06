import re

vocab = ['a', 'am', 'and', 'anywhere', 'are', 'be', 'boat', 'box', 'car',
         'could', 'dark', 'do', 'eat', 'eggs', 'fox', 'goat', 'good', 'green',
         'ham', 'here', 'house', 'i', 'if', 'in', 'let', 'like', 'may', 'me',
         'mouse', 'not', 'on', 'or', 'rain', 'sam', 'say', 'see', 'so', 'thank',
         'that', 'the', 'them', 'there', 'they', 'train', 'tree', 'try', 'will',
         'with', 'would', 'you']


doc = ("I would not like them here or there.\n"
       "I would not like them anywhere.\n"
       "I do not like green eggs and ham.\n"
       "I do not like them, Sam-I-am.")


doc = doc.lower()
doc = re.split("\\W", doc)
doc = [s for s in doc if len(s) > 0]


def get_bag_of_words(doc, vocab):

    word_count_dict = {word: 0 for word in vocab}

    for word in doc:
        if word in word_count_dict:
            word_count_dict[word] += 1

    bag = [0] * len(vocab)
    for i, word in enumerate(vocab):
        bag[i] = word_count_dict[word]
    
    return bag

# Get the bag of words for the document
bag_of_words = get_bag_of_words(doc, vocab)
print(bag_of_words)
