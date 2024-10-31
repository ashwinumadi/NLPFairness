import csv
from collections import defaultdict, Counter
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import spacy
import pickle

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load spaCy's English model for POS tagging
nlp = spacy.load("en_core_web_sm")

# Load BERT model and tokenizer, then move model to the selected device
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Function to get the embedding of a unigram in the context of a full sentence
def get_unigram_embedding_in_context(sentence, unigram):
    # Tokenize and prepare input, moving inputs to the GPU if available
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: [seq_length, hidden_size]
    
    # Tokenize unigram separately to locate its position within the sentence
    unigram_tokens = tokenizer.tokenize(unigram)
    unigram_token_ids = tokenizer.convert_tokens_to_ids(unigram_tokens)

    # Locate unigram within the sentence by matching token IDs
    sentence_token_ids = inputs['input_ids'].squeeze().tolist()
    for i in range(len(sentence_token_ids) - len(unigram_token_ids) + 1):
        if sentence_token_ids[i:i + len(unigram_token_ids)] == unigram_token_ids:
            # Move embedding to CPU and detach to get the numpy array
            unigram_embedding = embeddings[i:i + len(unigram_token_ids)].mean(dim=0).detach().cpu().numpy()
            return unigram_embedding

    # If unigram is not found, return a zero embedding
    return np.zeros(embeddings.size(-1))

# Step 1: Read sentences and filter unigrams that are adjectives or adverbs
unigram_embeddings = defaultdict(list)
unigram_counts = Counter()

# Replace 'your_file.csv' with your CSV file path
with open('/content/depression_dataset_reddit_cleaned.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        sentence = row[0].strip()
        
        # POS tagging using spaCy to find adjectives and adverbs
        doc = nlp(sentence)
        for token in doc:
            if token.pos_ in {"ADJ"}:  # Check if POS is adjective or adverb
                unigram = token.text
                unigram_counts[unigram] += 1  # Count the occurrence of each unigram
                unigram_embedding = get_unigram_embedding_in_context(sentence, unigram)
                unigram_embeddings[unigram].append(unigram_embedding)


# Display total number of unique unigrams and their individual counts
print(f"Total unique unigrams (adjectives and adverbs): {len(unigram_counts)}")
print("Individual unigram counts:")
for unigram, count in unigram_counts.items():
    print(f"{unigram}: {count}")

# Step 2: Calculate average embeddings for unigrams
average_unigram_embeddings = {}
for unigram, embeddings in unigram_embeddings.items():
    if len(embeddings) > 1:
        # Average embeddings if unigram appears more than once
        avg_embedding = np.mean(embeddings, axis=0)
    else:
        avg_embedding = embeddings[0]
    average_unigram_embeddings[unigram] = avg_embedding

# Output results: save to file
with open('unigram_dictionary.pkl', 'wb') as file:
    pickle.dump(average_unigram_embeddings, file)
