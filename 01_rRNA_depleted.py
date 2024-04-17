import json
import random
from sklearn.model_selection import train_test_split
import os

directory = os.get_cwd()
filename = 'rnacentral_dict.json'

with open(os.path.join(directory, filename), 'r') as f:
    data = json.load(f)

# randomly delete 66% of the 'rRNA' values
if 'rRNA' in data:
    rRNA_data = data['rRNA']
    rRNA_data = random.sample(rRNA_data, len(rRNA_data) // 3)
    data['rRNA'] = rRNA_data

# initialize the lists to hold train, test, and validation data
train_data = []
test_data = []
val_data = []
key_len_data = []

# split the data into train, test and validation
for key in data:
    if key == 'sgRNA':  # skip 'sgRNA'
        continue

    values = data[key]
    train, test = train_test_split(values, test_size=0.02, random_state=42)
    test, val = train_test_split(test, test_size=0.50, random_state=42)

    train_data.extend(train)
    test_data.extend(test)
    val_data.extend(val)

    key_len_data.append((key, len(values)))  # log the key and its values length

# shuffle the data
random.shuffle(train_data)
random.shuffle(test_data)
random.shuffle(val_data)

print(f'length of train is {len(train_data)}')
# save to files
with open(os.path.join(directory, 'train.txt'), 'w') as f:
    for item in train_data:
        f.write(f'{item}\n')
print(f'length of test is {len(test_data)}')
with open(os.path.join(directory, 'test.txt'), 'w') as f:
    for item in test_data:
        f.write(f'{item}\n')
print(f'length of val is {len(val_data)}')
with open(os.path.join(directory, 'val.txt'), 'w') as f:
    for item in val_data:
        f.write(f'{item}\n')

# write the key and length data to a separate file
with open(os.path.join(directory, 'key_length.txt'), 'w') as f:
    for key, length in key_len_data:
        f.write(f'{key}\n{length}\n')
