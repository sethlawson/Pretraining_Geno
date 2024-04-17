from Bio import SeqIO
import os
import json
from collections import defaultdict

def is_valid_rna(seq):
    for char in seq:
        if char not in ['A', 'T', 'G', 'C']:
            return False
    return True

directory = os.getcwd()
type_of_sequence_dict = defaultdict(list)

for filename in os.listdir(directory):
    if filename.endswith(".fasta"):
        for record in SeqIO.parse(os.path.join(directory, filename), "fasta"):
            seq = str(record.seq)
            seq = seq.replace('U', 'T')
            if is_valid_rna(seq):
                type_of_sequence = record.description.split(' ')[1]  # Extract the type of sequence
                type_of_sequence_dict[type_of_sequence].append(seq)

# Save the dictionary as a JSON file
with open(os.path.join(directory, 'rnacentral_dict.json'), 'w') as f:
    json.dump(type_of_sequence_dict, f)

# Save individual text files for each type of sequence
for type_of_sequence, sequences in type_of_sequence_dict.items():
    with open(os.path.join(directory, f'{type_of_sequence}.txt'), 'w') as f:
        for seq in sequences:
            f.write("%s\n" % seq)
