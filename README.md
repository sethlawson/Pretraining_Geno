# Pretraining_Geno
For pretraining ncRNA GPT large language models

# To train from scratch
Build the conda environment 'rna2' from the yaml file in the directory
Download rnacentral_active.fasta and place in the directory.
Enter the directory.
Execute python scripts 00-11.

00: Parses fasta headers for the rna species and generates a dictionary of sequences that are accessible by keys corresponding to the rna species. The dictionary is saved as rnacentral_dict.json. 

01: Loads rnacentral_dict.json, randomly downsamples the rRNA by 66%, then divides the data into a 98/1/1 train/test/val split stratified by rna species. Data is saved as train.txt, test.txt and val.txt.

02 & 06: Trains a tokenizer. Tokenizer saved to tokenizerN where N is the vocabulary size.

03 & 07: Tokenizes the test and train dataset and trains the model. Model checkpoints are saved to Geno_GPT2_N, where N is the vocabulary size. 

04 & 08: Loads rnacentral_dict.json, randomly selects up to 5000 RNAs of each species, tokenizes them and generates embeddings for use in script 11. Saved as embeddings_before_tsneN.json where N is the vocabulary size.

05 & 09: Generates 100 random 8 nt primer sequences, and then performs prompt completion using the pretrained model. Each of the 100 prompts is completed four times, three with a temperature of 1 and once with a temperature of 0.

10: Generates histogram of 500 token vocabulary word lengths.

11: Generates tSNE plots of the embeddings made in scripts 04 and 08.  

Scripts 03 and 07 require tailoring to your system. Batch sizes have been optimized for 48gb VRAM and can be adjusted as necessary. Note that the scripts specify the CUDA devices explicitly (0 and 1 for scripts 03 and 07, respectively). 
