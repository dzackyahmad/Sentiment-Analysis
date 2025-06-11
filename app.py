import streamlit as st
import pandas as pd
from pipeline.lexicon_classifier import load_lexicon, calculate_sentiment, classify_sentiment
from pipeline.bert_predictor import load_model as load_bert_model, predict_sentiment as predict_bert
from pipeline.preprocessing import preprocess_text
from utils.style import apply_custom_css

# === SETUP ===
st.set_page_config(page_title="Sentiment App", layout="centered")
st.title("🧠 Analisis Sentimen – Bahasa Indonesia")
apply_custom_css()

# === METODE ===
method = st.selectbox(
    "Pilih metode analisis:",
    ["Lexicon Based", "BERT Based"]
)

# === INPUT ===
st.subheader("Masukkan teks ulasan:")
with st.expander("📌 Contoh kalimat yang bisa kamu coba"):
    st.markdown("""
    - *Aplikasinya cepat dan sangat membantu!*
    - *Aplikasinya jelek banget, sering error, ileggal lagi.*
    - *Biasa aja sih, nggak terlalu spesial.*
    """)

user_input = st.text_area("Tulis ulasan Anda di sini...",)


# === LEXICON ===
@st.cache_data
def load_all_lexicons():
    pos = load_lexicon("lexicon/positive.tsv")
    neg = load_lexicon("lexicon/negative.tsv")
    return {**pos, **neg}

# === ANALISIS ===
if st.button("🔍 Analisis Sentimen"):
    if not user_input.strip():
        st.warning("Teks ulasan tidak boleh kosong.")
    else:
        st.markdown("### 🔎 Hasil Analisis:")

        if method == "Lexicon Based":
            lexicon = load_all_lexicons()
            tokens = preprocess_text(user_input)
            score = calculate_sentiment(tokens, lexicon)
            label = classify_sentiment(score)

            st.write(f"**Metode:** Lexicon")
            st.write(f"**Label Sentimen:** `{label}`")
            st.write(f"**Skor Sentimen:** `{score}`")
            st.write(f"**Token yang dikenali:** `{tokens}`")

        elif method == "BERT Based":
            bert_input = user_input
            try:
                with st.spinner("Memuat model BERT..."):
                    model, tokenizer = load_bert_model("saved_model")
                    pred_label, confidence = predict_bert(bert_input, model, tokenizer)
                st.success(f"Prediksi sentimen: {pred_label} (Confidence: {confidence:.2f})")
            except Exception as e:
                st.error(f"Gagal memuat model BERT: {str(e)}")

# === OPTIONAL CHART ===
st.divider()
st.subheader("📈 Distribusi Sentimen (dari dataset)")

try:
    df = pd.read_csv("data/with_labels.csv")
    sent_count = df["sentiment_label"].value_counts()
    st.bar_chart(sent_count)
except:
    st.info("Belum ada data klasifikasi di `with_labels.csv`.")
