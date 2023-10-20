import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import numpy as np

# Load pre-trained model and tokenizer from transformers
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

def predict_sentiment(text):
    # Tokenize input text and convert to tensor
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Calculate probabilities using softmax and get the predicted label
    probs = softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probs, dim=1).item()
    
    return predicted_label, probs.numpy()

# Example
text = "I absolutely loved the customer service!"
label, probs = predict_sentiment(text)
sentiments = ["very negative", "negative", "neutral", "positive", "very positive"]
print(f"The review is {sentiments[label]} with probabilities: {probs}")
