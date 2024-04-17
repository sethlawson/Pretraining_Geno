from Bio import SeqIO
from sklearn.model_selection import train_test_split
from tokenizers import CharBPETokenizer
import os
import math

paths = ["val.txt"]

# Initialize a tokenizer
tokenizer = CharBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=4, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
if not os.path.isdir('tokenizer4'):
    os.mkdir('tokenizer4')
tokenizer.save_model('tokenizer4')


