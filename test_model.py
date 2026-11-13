import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model/best_model"

print("Loading ToxicGuard AI...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

print("Model loaded successfully!")

text = "You are an absolutely disgusting person."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=256
)

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs.logits)

labels = [
    "Toxic",
    "Severe Toxic",
    "Obscene",
    "Threat",
    "Insult",
    "Identity Hate"
]

print("\nComment:", text)
print("\nPredictions:")

for label, probability in zip(labels, probabilities[0]):
    print(f"{label}: {probability.item():.4f}")