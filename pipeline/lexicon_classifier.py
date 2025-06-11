# pipeline/lexicon_classifier.py

import os
import pandas as pd
import ast
import csv

def load_lexicon(path):
    """Load file .tsv dengan format: word \t score"""
    lexicon = {}
    with open(path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        header = next(reader)  # Lewati header
        for row in reader:
            try:
                word = row[0].strip()
                score = int(row[1])
                lexicon[word] = score
            except (IndexError, ValueError):
                continue
    return lexicon

def calculate_sentiment(tokens, lexicon):
    """Hitung total skor sentimen dari token-token yang ditemukan dalam lexicon"""
    return sum(lexicon.get(token, 0) for token in tokens)

def classify_sentiment(score):
    """Konversi skor total menjadi label sentimen"""
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    return "Neutral"

def run_lexicon_classifier(
    input_path="data/processed_file.csv",
    pos_path="lexicon/positive.tsv",
    neg_path="lexicon/negative.tsv",
    output_path="data/with_labels.csv"
):
    print(f"📖 Membaca data dari: {input_path}")
    df = pd.read_csv(input_path)

    df['processed_text'] = df['processed_text'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    print("🔍 Memuat lexicon positif & negatif...")
    positive = load_lexicon(pos_path)
    negative = load_lexicon(neg_path)
    lexicon = {**positive, **negative}

    print("⚙️ Menghitung skor & label...")
    df['sentiment_score'] = df['processed_text'].apply(lambda tokens: calculate_sentiment(tokens, lexicon))
    df['sentiment_label'] = df['sentiment_score'].apply(classify_sentiment)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/with_scores.csv", index=False)
    df.to_csv(output_path, index=False)

    print(f"✅ Hasil disimpan ke: {output_path}")
