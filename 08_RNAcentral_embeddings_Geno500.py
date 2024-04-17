import os
from transformers import GPT2Model, GPT2TokenizerFast, Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from transformers import TrainerCallback
from transformers import IntervalStrategy
import json
import random
from transformers import GPT2Model, GPT2TokenizerFast

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_path = "Geno_GPT2_500"  # Updated path to the GPT2 model
tokenizer_path = "tokenizer500"  # Updated path to the tokenizer
model = GPT2Model.from_pretrained(model_path)
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
tokenizer.pad_token = '<pad>'



# Load RNA sequences from the dictionary file
with open("rnacentral_dict.json", "r") as f:
    rna_dict = json.load(f)


# Define a function to tokenize sequences and obtain last hidden state embeddings
def get_embeddings(sequences):
    # Initialize an empty list to store the embeddings
    all_embeddings = []
    
    # Iterate over each sequence
    for sequence in sequences:
        # Tokenize the sequence
        encoded_seq = tokenizer(sequence, truncation=True, padding=True, return_tensors="pt")
        
        # Check if the input sequence length exceeds the maximum length supported by the model
        if len(encoded_seq['input_ids'][0]) > 1024:
            # Truncate the sequence if it exceeds the maximum length
            encoded_seq['input_ids'] = encoded_seq['input_ids'][:, :1024]
            encoded_seq['attention_mask'] = encoded_seq['attention_mask'][:, :1024]
        
        # Obtain the hidden states
        with torch.no_grad():
            outputs = model(encoded_seq['input_ids'])
            hidden_states = outputs[0]
        
        # Initialize an empty list to store Geno embeddings
        geno_embeddings = []

        
        averaged_embedding = torch.mean(hidden_states, dim=1)

        all_embeddings.append(averaged_embedding)

    
    # Concatenate the embeddings along the first axis
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    return embeddings

# Initialize lists to store embeddings and corresponding labels
embeddings = []
labels = []

# Initialize a dictionary to store embeddings before dimensionality reduction
embeddings_dict = {}

# Iterate over RNA species in the dictionary
for species, sequences in rna_dict.items():
    print(species)
    # Randomly sample 5k sequences if available, otherwise use all available sequences
    sampled_sequences = random.sample(sequences, min(len(sequences), 5000))
    
    # Tokenize sequences and obtain embeddings
    species_embeddings = get_embeddings(sampled_sequences)
    
    # Save the embeddings for the species
    embeddings_dict[species] = species_embeddings
    
    # Append labels
    labels.extend([species] * len(species_embeddings))

# Save the embeddings dictionary to a file
output_path = "embeddings_before_tsne500.json"

embeddings_dict_serializable = {key: value.tolist() for key, value in embeddings_dict.items()}

# Write the dictionary to the JSON file
with open(output_path, "w") as f:
    json.dump(embeddings_dict_serializable, f)

