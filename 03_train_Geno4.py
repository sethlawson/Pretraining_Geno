from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, GPT2Config
from transformers import TrainerCallback, TrainerState, TrainerControl
from datasets import load_dataset, load_from_disk
import os
import random
import glob
import logging

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("log4.txt"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class LogLossCallback(TrainerCallback):
    "A callback that logs the training loss every `logging_steps` steps."
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % args.logging_steps == 0 and state.global_step > 0:
            try:
                logs = {'loss': state.log_history[-1]['loss']}
            except KeyError:
                logs = state.log_history[-1]
            for key, value in logs.items():
                logger.info(f"{key}: {value}")

# For using a specific device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer4")

# Set the padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = "<pad>"

# Configure the training arguments
training_args = TrainingArguments(
    output_dir="Geno_GPT2_4",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='log4',
    logging_steps=500,
    save_steps=5000,
    evaluation_strategy="steps",
    eval_steps=5000,
    dataloader_num_workers=16,
    log_level='debug',
    gradient_accumulation_steps=1,
    fp16=True,  
    fp16_opt_level='O1',  
    dataloader_drop_last=True,  
)


# Check if the tokenized datasets exist
if not os.path.exists("tokenized_train_dataset4PT") or not os.path.exists("tokenized_test_dataset4PT"):
    # Load the training and test datasets
    train_dataset = load_dataset('text', data_files='train.txt') #??['train']
    test_dataset = load_dataset('text', data_files='test.txt')
    # Tokenize the datasets
    def tokenize_function(examples):
        return {
            "input_ids": tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)[
                "input_ids"],
            "labels": tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)["input_ids"],
        }
    # Increase the number of processes for tokenizing the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=16, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=16, remove_columns=["text"])

    # Save the tokenized datasets
    train_dataset.save_to_disk("tokenized_train_dataset4PT")
    test_dataset.save_to_disk("tokenized_test_dataset4PT")
else:
    # Load the tokenized datasets
    train_dataset = load_from_disk("tokenized_train_dataset4PT")
    test_dataset = load_from_disk("tokenized_test_dataset4PT")

# Create a GPT-2 configuration
config = GPT2Config(
    vocab_size=tokenizer.vocab_size, # Size of the vocabulary
    bos_token_id=tokenizer.bos_token_id, # The id of the beginning-of-sentence token
    eos_token_id=tokenizer.eos_token_id, # The id of the end-of-sentence token
    pad_token_id=tokenizer.pad_token_id,
)

# Check if there are any checkpoints
checkpoints = list(os.path.join(c) for c in sorted(glob.glob(training_args.output_dir + "/checkpoint-*"), key=lambda x: int(x.split('-')[-1])))
print(checkpoints)

# If there are checkpoints, load the model from the last checkpoint, otherwise initialize a new one
if checkpoints:
    model = GPT2LMHeadModel.from_pretrained(checkpoints[-1])
else:
    model = GPT2LMHeadModel(config)

# Initialize the Trainer
trainer = Trainer(
    model=model, # The instantiated 🤗 Transformers model to be trained
    args=training_args, # Training arguments
    train_dataset=train_dataset, # Training dataset
    eval_dataset=test_dataset, # Evaluation dataset
    callbacks=[LogLossCallback],
)

print("Tokenizer pad token: ", tokenizer.pad_token, " ID: ", tokenizer.pad_token_id)
print("Model configuration: ", model.config)
print("TrainingArguments: ", training_args)

# Get first example from the training dataset
data = train_dataset[0]

# Fetch input_ids and attention_mask from the example
input_ids = data['input_ids']  # get the input_ids

# Decode
decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)

print(f'Decoded text: {decoded_text}')
print(f'Input IDs: {input_ids}')

# If there are checkpoints, resume training from the last checkpoint, otherwise start training from scratch
if checkpoints:
    trainer.train(resume_from_checkpoint=checkpoints[-1])
else:
    trainer.train()
# Save the model
model.save_pretrained("Geno_GPT2_4")
