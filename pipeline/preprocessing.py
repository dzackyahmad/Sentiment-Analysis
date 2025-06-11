# pipeline/preprocessing.py

import os
import re
import string
import pandas as pd
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Unduh data NLTK jika belum ada
nltk.download('punkt_tab')
nltk.download('stopwords')

# Inisialisasi Stemmer dan Stopwords Bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stopwords_id = set(stopwords.words('indonesian'))

def preprocess_text(text):
    """
    Membersihkan teks:
    - Lowercase
    - Hapus angka dan tanda baca
    - Tokenisasi
    - Hapus stopwords
    - Stemming
    """
    # Lowercase
    text = text.lower()
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    # Hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenisasi
    tokens = word_tokenize(text)
    # Hapus stopwords
    tokens = [t for t in tokens if t not in stopwords_id]
    # Stemming
    stemmed = [stemmer.stem(t) for t in tokens]
    return stemmed

def clean_data(input_path='data/raw_reviews.csv'):
    print(f"📖 Membaca file: {input_path}")
    df = pd.read_csv(input_path)

    if 'ulasan' not in df.columns:
        raise ValueError("❌ Kolom 'ulasan' tidak ditemukan di file!")

    # Proses semua teks
    print("⚙️ Preprocessing semua ulasan...")
    df['processed_text'] = df['ulasan'].astype(str).apply(preprocess_text)

    # Gabungkan token jadi string bersih untuk keperluan visual/manual
    df['cleaned'] = df['processed_text'].apply(lambda x: ' '.join(x))

    # Buat folder data kalau belum ada
    os.makedirs("data", exist_ok=True)

    # Simpan hasil
    df[['ulasan', 'cleaned']].to_csv('data/cleaned.csv', index=False)
    df[['ulasan', 'processed_text']].to_csv('data/processed_file.csv', index=False)
    print("✅ Preprocessing selesai. Disimpan ke:")
    print("  → data/cleaned.csv")
    print("  → data/processed_file.csv")
