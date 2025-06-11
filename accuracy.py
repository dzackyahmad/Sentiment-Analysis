import pandas as pd
# Hapus import AutoModelForSequenceClassification
from transformers import AutoTokenizer # Tetap pakai AutoTokenizer jika Anda ingin fleksibilitas tokenizer
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

# Import fungsi load_model dan predict_sentiment dari pipeline/bert_predictor.py
from pipeline.bert_predictor import load_model, predict_sentiment # <--- Tambahkan ini
# Perhatikan bahwa nama fungsinya sekarang load_model, bukan load_bert_model seperti di app streamlit

# --- 1. Memuat Data ---
try:
    df = pd.read_csv('data/with_scores.csv')
    print("Data 'data/with_scores.csv' berhasil dimuat.")
except FileNotFoundError:
    print("Error: File 'data/with_scores.csv' tidak ditemukan. Pastikan file berada di direktori yang benar.")
    exit()

# --- 2. Menyiapkan Data ---
texts = df['ulasan'].tolist()
true_labels_text = df['sentiment_label'].tolist()

# --- 3. Memuat Model dan Tokenizer ---
model_path = "./saved_model" # Path ke direktori model yang disimpan

try:
    # GUNAKAN FUNGSI LOAD_MODEL DARI bert_predictor.py ANDA!
    model, tokenizer = load_model(model_path) # <--- Ubah baris ini
    print("Model dan Tokenizer berhasil dimuat menggunakan load_model dari bert_predictor.")
except Exception as e:
    print(f"Error saat memuat model atau tokenizer dari '{model_path}': {e}")
    print("Pastikan direktori 'saved_model' ada dan berisi file yang diperlukan (pytorch_model.bin, tokenizer_config.json, vocab.txt, dll.).")
    exit()

# Menentukan pemetaan label (label_mapping)
# Ini harus sesuai dengan label_map di fungsi predict_sentiment Anda (0: Negative, 1: Neutral, 2: Positive)
label_mapping = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}
id_to_label = {v: k for k, v in label_mapping.items()} # Untuk laporan klasifikasi
print(f"Menggunakan pemetaan label: {label_mapping}")


# Mengkonversi label kebenaran menjadi ID numerik
try:
    true_labels_ids = [label_mapping[label] for label in true_labels_text]
    print(f"Label kebenaran (5 pertama, dalam ID): {true_labels_ids[:5]}")
except KeyError as e:
    print(f"\nError: Label '{e}' dari 'sentiment_label' di CSV tidak ditemukan dalam pemetaan label model.")
    print("Pastikan semua nilai di kolom 'sentiment_label' CSV Anda sudah ada di dalam 'label_mapping'.")
    print(f"Label unik di data Anda: {df['sentiment_label'].unique()}")
    print(f"Label di pemetaan yang digunakan: {list(label_mapping.keys())}")
    exit()


# --- 4. Mengatur Device (CPU/GPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Mengatur model ke mode evaluasi (penting untuk inferensi)
print(f"Menggunakan device: {device}")

# --- 5. Fungsi untuk Melakukan Inferensi (Prediksi) ---
# Gunakan fungsi predict_sentiment yang sudah ada di bert_predictor.py
def predict_batch(texts, model, tokenizer, device, batch_size=16):
    predicted_labels_ids = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        for text in batch_texts:
            # predict_sentiment mengembalikan label teks dan confidence
            # Kita hanya butuh label ID untuk metrik
            pred_label_text, _ = predict_sentiment(text, model, tokenizer)
            predicted_labels_ids.append(label_mapping[pred_label_text])
    return predicted_labels_ids


# Melakukan prediksi pada seluruh dataset
print("\nMelakukan prediksi pada data...")
predicted_labels_ids = predict_batch(texts, model, tokenizer, device)
print(f"Label prediksi (5 pertama, dalam ID): {predicted_labels_ids[:5]}")

# --- 6. Menghitung Metrik ---
print("\n--- Hasil Evaluasi Model ---")

# Menghitung Accuracy
accuracy = accuracy_score(true_labels_ids, predicted_labels_ids)
print(f"Accuracy: {accuracy:.4f}")

# Menghitung F1-score
f1_weighted = f1_score(true_labels_ids, predicted_labels_ids, average='weighted')
f1_macro = f1_score(true_labels_ids, predicted_labels_ids, average='macro')

print(f"F1-score (Weighted): {f1_weighted:.4f}")
print(f"F1-score (Macro): {f1_macro:.4f}")

# Menampilkan Laporan Klasifikasi yang lebih detail
target_names = [id_to_label[i] for i in sorted(id_to_label.keys())]

print("\nLaporan Klasifikasi Detail:")
print(classification_report(true_labels_ids, predicted_labels_ids, target_names=target_names))