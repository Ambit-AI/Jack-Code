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
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.to(device)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments and Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_steps=int(1e9),  # set to a very high number to avoid logging
    learning_rate=3e-5,
    num_train_epochs=5,
    output_dir="./results",        
    logging_dir=None,       # set to None to avoid logging
    gradient_accumulation_steps=2,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Train the model
trainer.train()

# Define the prediction function
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_label = torch.argmax(probs, dim=1).item()

    sentiments = ["negative", "positive"]
    return sentiments[predicted_label], probs.cpu().numpy()

# Test
test_reviews = [
    "The food was absolutely wonderful, from preparation to presentation, very pleasing.",
    "I did not enjoy my experience at this restaurant. The service was slow and the food was cold."
]

for review in test_reviews:
    sentiment, probability = predict_sentiment(review)
    print(f"Review: {review}\nPredicted Sentiment: {sentiment} with Probabilities: {probability}\n")
