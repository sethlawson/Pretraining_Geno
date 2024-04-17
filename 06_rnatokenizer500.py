from Bio import SeqIO
from sklearn.model_selection import train_test_split
from tokenizers import CharBPETokenizer
import os
import math


paths = ["val.txt"]

# Initialize a tokenizer
tokenizer = CharBPETokenizer()

# Customize training
tokenizer.train(files=path, vocab_size=500, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
if not os.path.isdir('tokenizer500'):
    os.mkdir('tokenizer500')
tokenizer.save_model('tokenizer500')
