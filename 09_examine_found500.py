from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import os
import csv
import random

# Define the DNA nucleotides
nucleotides = ['A', 'C', 'G', 'T']

# Function to generate a random DNA sequence
def generate_random_sequence(length):
    return ''.join(random.choice(nucleotides) for _ in range(length))

# Define the model and tokenizer path
model_path = "Geno_GPT2_500"
tokenizer_path = "tokenizer500"

# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)

# Function to generate text
def generate_text(input_sequence, deterministic=False):
    # Encode the input sequence
    input_ids = tokenizer.encode(input_sequence, return_tensors='pt').to(device)

    # Generate text
    if deterministic:
        output = model.generate(input_ids, max_length=1024)
    else:
        output = model.generate(input_ids, max_length=1024, temperature=1, num_return_sequences=3, do_sample=True)

    # Decode the output
    generated_texts = [tokenizer.decode(out, skip_special_tokens=True) for out in output]

    return generated_texts

# Generate 100 random 8nt long sequences
sequences = [generate_random_sequence(8) for _ in range(100)]

# Define conditions
conditions = ['random', 'deterministic']

# Generate completions and write results for each condition
for condition in conditions:
    # Write the results to a CSV file
    with open(f'sequence_completions_{condition}500.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Prompt", "Completion 1", "Completion 2", "Completion 3"])

        # Generate completions for each sequence
        for sequence in sequences:
            if condition == 'deterministic':
                generated_text = generate_text(sequence, True)
                writer.writerow([sequence, generated_text[0], generated_text[0], generated_text[0]])
            else:
                generated_texts = generate_text(sequence)
                writer.writerow([sequence] + generated_texts)
