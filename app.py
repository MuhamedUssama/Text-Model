import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, request, jsonify
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os
import urllib.request

print("Starting app.py")

# إعداد بيانات NLTK
nltk.download('punkt')
nltk.download('stopwords')
print("NLTK data downloaded")

app = Flask(__name__)
print("Flask app created")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

trait_info = [
    ('A', 'Agreeableness(A)'),
    ('C', 'Conscientiousness(C)'),
    ('E', 'Extraversion(E)'),
    ('N', 'Neuroticism(N)'),
    ('O', 'Openness(O)')
]
trait_columns = [info[0] for info in trait_info]

print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Tokenizer loaded")

print("Loading model...")
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(trait_columns),
    problem_type="multi_label_classification"
)
print("Model initialized")

model.config.hidden_dropout_prob = 0.4
model.config.attention_probs_dropout_prob = 0.4

model.to(device)
print("Model moved to device")

# رابط Azure Blob Storage لتحميل النموذج
model_url = 'https://traitmodelstore.blob.core.windows.net/models/Bert_person_improve.pth?sp=r&st=2025-04-19T15:19:56Z&se=2026-06-30T22:19:56Z&spr=https&sv=2024-11-04&sr=b&sig=XugrFOLosZwvgrZR4cEyITrDbOHBE6DP62m4IFV2uEk%3D'
model_local_path = 'Bert_person_improve.pth'

try:
    if not os.path.exists(model_local_path):
        print("Downloading model from Azure Blob Storage...")
        urllib.request.urlretrieve(model_url, model_local_path)
        print("Download complete.")
    print("Loading model state dict...")
    model.load_state_dict(torch.load(model_local_path, map_location=device))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# لو مش عندك thresholds، هنستخدم 0.5
thresholds_path = 'best_thresholds.npy'
if os.path.exists(thresholds_path):
    best_thresholds = np.load(thresholds_path)
    print(f"Using thresholds: {best_thresholds}")
else:
    best_thresholds = [0.5] * len(trait_columns)
    print("Warning: Using default thresholds [0.5].")

model.eval()
print("Model set to evaluation mode")

def preprocess_input_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def analyze_personality(text, model, tokenizer, trait_info, device, thresholds):
    cleaned_text = preprocess_input_text(text)
    encoding = tokenizer(
        cleaned_text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()[0]

    probabilities = torch.sigmoid(torch.tensor(logits)).numpy() * 100
    binary_preds = [(probabilities[i] / 100 > thresholds[i]) for i in range(len(trait_info))]
    result = {
        full_name: f"{prob:.2f}%"
        for (_, full_name), prob in zip(trait_info, probabilities)
    }
    return result, binary_preds, probabilities

@app.route('/analyze_personality', methods=['POST'])
def analyze_personality_api():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        user_input = data['text']
        if not user_input.strip():
            return jsonify({"error": "Input text cannot be empty"}), 400
        result, binary_preds, probabilities = analyze_personality(user_input, model, tokenizer, trait_info, device, best_thresholds)
        dominant_trait = None
        if any(binary_preds):
            max_trait_idx = np.argmax(probabilities)
            dominant_trait = trait_info[max_trait_idx][1]

        response = {
            "dominant_trait": dominant_trait if dominant_trait else "No dominant trait detected",
            "traits": result
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)