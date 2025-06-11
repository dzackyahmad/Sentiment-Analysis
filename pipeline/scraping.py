# pipeline/scraping.py

import os
import pandas as pd
from google_play_scraper import Sort, reviews

def scrape_reviews(app_id='com.gojek.app', total=3000, save_path='data/raw_reviews.csv'):

    print(f"🚀 Memulai scraping {total} review dari: {app_id}")
    batch_size = 200  # Maksimal per request (dari library)
    all_reviews = []
    continuation_token = None

    while len(all_reviews) < total:
        count = min(batch_size, total - len(all_reviews))  # Ambil sesuai sisa
        result, continuation_token = reviews(
            app_id,
            lang='id',                # Bahasa Indonesia
            country='id',             # Lokasi Indonesia
            sort=Sort.MOST_RELEVANT,  # Urutkan berdasarkan relevansi
            count=count,
            continuation_token=continuation_token
        )

        if not result:
            print("‼️ Tidak ada review tambahan. Berhenti.")
            break

        all_reviews.extend(result)
        print(f"✅ Total review terkumpul: {len(all_reviews)}")

        # Jika tidak ada token lanjutan, berarti data habis
        if continuation_token is None:
            break

    if not all_reviews:
        print("‼️ Gagal mengambil data.")
        return

    # Ambil kolom yang kita butuhkan: isi dan skor
    df = pd.DataFrame(all_reviews)[['content', 'score']]
    df.columns = ['ulasan', 'label']

    # Buat folder kalau belum ada
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Simpan ke CSV
    df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"📁 Review berhasil disimpan ke: {save_path}")
