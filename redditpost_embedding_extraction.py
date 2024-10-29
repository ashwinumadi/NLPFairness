import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import csv
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import pickle

unique_sentences = set()

with open('/content/depression_dataset_reddit_cleaned.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        sentence = row[0].strip()
        unique_sentences.add(sentence)

unique_sentences_list = list(unique_sentences)

# Tokenize and clean
stop_words = set(stopwords.words('english'))
all_sentences = ' '.join(unique_sentences_list)
words = [word for word in word_tokenize(all_sentences.lower()) if word.isalpha() and word not in stop_words]


# Count frequencies

trigrams = Counter(ngrams(words, 3))

most_common = trigrams.most_common(10000)

list_most_common = [' '.join(i[0]) for i in most_common]


with open('trigram_dictionary.pkl', 'rb') as file:
    original_dict = pickle.load(file)

key_list = list_most_common  

filtered_dict = {key: original_dict[key] for key in key_list if key in original_dict}

# Save the filtered dictionary to a new pickle file
with open('filtered_dict.pkl', 'wb') as file:
    pickle.dump(filtered_dict, file)
