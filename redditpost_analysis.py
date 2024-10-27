import csv
from nltk.util import ngrams
from collections import Counter, defaultdict
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get the embedding of a trigram in the context of a full sentence
def get_trigram_embedding_in_context(sentence, trigram):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: [seq_length, hidden_size]
    
    # Tokenize trigram separately to locate its position within the sentence
    trigram_tokens = tokenizer.tokenize(trigram)
    trigram_token_ids = tokenizer.convert_tokens_to_ids(trigram_tokens)
    #print(trigram_token_ids)
    # Locate trigram within the sentence by matching token IDs
    sentence_token_ids = inputs['input_ids'].squeeze().tolist()
    #print(sentence_token_ids)
    # Find the starting position of the trigram in the sentence
    for i in range(len(sentence_token_ids) - len(trigram_token_ids) + 1):
        if sentence_token_ids[i:i + len(trigram_token_ids)] == trigram_token_ids:
            trigram_embedding = embeddings[i:i + len(trigram_token_ids)].mean(dim=0).detach().numpy()
            return trigram_embedding

    # If trigram is not found (rare), return a zero embedding
    return np.zeros(embeddings.size(-1))

# Step 1: Read sentences and generate trigrams
trigram_counts = Counter()
trigram_embeddings = defaultdict(list)

# Replace 'your_file.csv' with your CSV file path
with open('depression_dataset_reddit_cleaned.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        sentence = row[0].strip()
        
        # Generate trigrams for the sentence
        tokens = sentence.split()  # Tokenize by whitespace
        trigrams = list(ngrams(tokens, 3))
        
        # Count trigrams and collect embeddings
        for trigram in trigrams:
            trigram_text = ' '.join(trigram)
            trigram_counts[trigram_text] += 1
            
            # Get trigram embedding in context
            embedding = get_trigram_embedding_in_context(sentence, trigram_text)
            trigram_embeddings[trigram_text].append(embedding)

# Step 2: Calculate average embeddings for trigrams
average_embeddings = {}
for trigram, embeddings in trigram_embeddings.items():
    if len(embeddings) > 1:
        # Average embeddings if trigram appears more than once
        avg_embedding = np.mean(embeddings, axis=0)
    else:
        avg_embedding = embeddings[0]
    average_embeddings[trigram] = avg_embedding

# Output results: each trigram with its count and average embedding
#for trigram, count in trigram_counts.items():
#    print(f"Trigram: {trigram}, Count: {count}, Average Embedding: {average_embeddings[trigram]}")
import pickle
with open('trigram_dictionary.pkl', 'wb') as file:  # 'wb' means write in binary mode
    pickle.dump(average_embeddings, file)