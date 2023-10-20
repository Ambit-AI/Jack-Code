import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from tqdm import tqdm

# Checking CUDA availability
print(torch.cuda.is_available())

# Set device
device = torch.device("cuda:0")

# Load dataset
dataset = load_dataset("yelp_polarity")

# Load pre-trained model and tokenizer
model_name = "bert-large-uncased"  # Use a larger BERT model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.to(device)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)  # Increased max_length

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments and Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=8,    # Reduced due to the larger model
    per_device_eval_batch_size=8,    # Reduced due to the larger model
    evaluation_strategy="steps",
    eval_steps=500,    # Reduced eval_steps for more frequent evaluation
    logging_steps=int(1e9),
    learning_rate=2e-5,    # Adjusted learning rate
    num_train_epochs=3,    # Reduced epochs due to potential overfitting with more epochs
    output_dir="./results",
    logging_dir=None,
    gradient_accumulation_steps=2,
    warmup_steps=500,    # Added warmup
    weight_decay=0.01,   # Weight decay for regularization
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Train the model
trainer.train()
