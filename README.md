# 🧠 Sentiment Analysis – Bahasa Indonesia (Lexicon & BERT Based)

Aplikasi web berbasis **Streamlit** untuk menganalisis sentimen teks ulasan aplikasi dalam Bahasa Indonesia. Mendukung dua metode: **Lexicon-Based** dan **BERT-Based**.

---

## 🚀 Fitur Utama

- Analisis sentimen dari input teks ulasan.
- Dua metode analisis:
  - ✅ Lexicon-based (berbasis kamus kata positif/negatif)
  - 🤖 BERT-based (deep learning dengan model BERT)
- Preprocessing teks menggunakan NLTK + Sastrawi
- Visualisasi distribusi sentimen
- Tampilan UI modern dan responsif

---

## 🔄 Alur Kerja Aplikasi

1. **Scraping Ulasan**
   - Mengambil data review dari Play Store (misalnya: Gojek)
   - Disimpan ke `data/raw_reviews.csv`

2. **Preprocessing**
   - Membersihkan teks: lowercase, hapus angka, tanda baca, stopword, stemming
   - Simpan hasil ke `data/cleaned.csv`

3. **Labeling dengan Lexicon**
   - Hitung skor sentimen dengan kamus kata (`positive.tsv`, `negative.tsv`)
   - Tentukan label (`Positive`, `Neutral`, `Negative`)
   - Simpan ke `data/with_labels.csv`

4. **Training Model BERT**
   - Load data dari `cleaned.csv` dan `with_labels.csv`
   - Train model BERT (`bert-base-uncased`)
   - Simpan model ke `saved_model/` dan tokenizer ke `models/bert_sentiment/`

5. **Aplikasi Web**
   - User masukkan teks ulasan
   - Pilih metode analisis
   - Tampilkan hasil sentimen dan confidence
   - Visualisasi chart distribusi sentimen dari dataset

---

## 🧪 Cara Menjalankan

### 1. **Clone dan Install**
```bash
git clone https://github.com/nama-kamu/Sentiment-Analysis.git
cd Sentiment-Analysis
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
pip install -r requirements.txt

### 2. **Scrapping dan Prepocessing**
python pipeline/scraping.py
python pipeline/preprocessing.py
python pipeline/lexicon_classifier.py

### 3. Training Model BERT
python pipeline/bert_trainer.py

### 4. Jalankan Aplikasi
streamlit run app.py
