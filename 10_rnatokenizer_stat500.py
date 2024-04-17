from tokenizers import ByteLevelBPETokenizer
from collections import Counter
import matplotlib.pyplot as plt

# Load tokenizer
tokenizer = ByteLevelBPETokenizer("tokenizer500/vocab.json", "tokenizer500/merges.txt")

# Get vocab
vocab = tokenizer.get_vocab()

# Print the size of the vocabulary
print(f"Vocabulary size: {len(vocab)}")

# Distribution of lengths of the words
lengths = [len(word) for word in vocab.keys()]
lengths_counter = Counter(lengths)
print(lengths_counter.most_common(1)[0][0])
print(lengths_counter)

# Plot distribution
plt.figure(figsize=(10,6))
plt.bar(lengths_counter.keys(), lengths_counter.values())
plt.xlabel("Word Length")
plt.ylabel("Frequency")
plt.title("Distribution of Word Lengths in Vocabulary")
plt.savefig('Figures/distribution500.svg')
plt.savefig('Figures/distribution500.png')
