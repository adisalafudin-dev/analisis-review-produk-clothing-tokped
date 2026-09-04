import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

INPUT_FILE = "reviews_tokopedia_clean.csv"
OUTPUT_FILE = "dataset_labeled.csv"

def beri_label(rating):
    """Rating 1-3 -> buruk, rating 4-5 -> baik."""
    if rating >= 4:
        return "baik"
    else:  # rating 1, 2, atau 3
        return "buruk"
    

def main():
    data_dict = {}
    df = pd.read_csv(INPUT_FILE)
    
    #Jumlah Data Masuk
    count_data_masuk = f"Data masuk: {len(df)} baris"
    data_dict["count_data_masuk"] = count_data_masuk
    
    # Buang baris yang rating-nya kosong/tidak valid (tidak bisa dilabel)
    sebelum = len(df)
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)
    df = df[df["rating"].between(1, 5)]
    sesudah = len(df)
    if sebelum != sesudah:
        count_baris_dibuang = f"{sebelum - sesudah} baris dibuang (rating kosong/tidak valid)"
        data_dict["count_baris_dibuang"] = count_baris_dibuang
        
    df["status"] = df["rating"].apply(beri_label)
    
    count_status = df["status"].value_counts()
    presentase_status = (df["status"].value_counts(normalize=True) * 100).round(1)
    
    data_dict["jumlah_status"] = count_status
    data_dict["presentase_status"] = presentase_status
        
    count_status_per_brand = pd.crosstab(df["brand"], df["status"])
    data_dict["status_per_brand"] = count_status_per_brand    
    
    # Distribusi rating mentah per brand (buat sanity check)

    
    count_rating_per_brand = pd.crosstab(df["brand"], df["rating"])
    data_dict["rating_per_brand"] = count_rating_per_brand
    
    # Simpan kolom final: text, status, brand (rating asli tetap disimpan
    # buat referensi/validasi, sesuai pola proyek sebelumnya)
    df_final = df[["text", "rating", "status", "brand"]]
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
 
    print(f"\nTersimpan di '{OUTPUT_FILE}'")
 
    # Catatan soal skenario 1000 data (500 baik, 500 buruk) dari roadmap
    jumlah_baik = (df["status"] == "baik").sum()
    jumlah_buruk = (df["status"] == "buruk").sum()

    data_dict["count_good_reviews"] = jumlah_baik
    data_dict["count_bad_reviews"] = jumlah_buruk
    
    table_data = [["Key", "Value"]] + [[str(k), str(v)] for k, v in data_dict.items()]
    doc = SimpleDocTemplate("dictionary_output.pdf", pagesize=letter)
    pdf_table = Table(table_data)
    
    pdf_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    
    doc.build([pdf_table])

    if jumlah_baik < 500 or jumlah_buruk < 500:
        kurang = "baik" if jumlah_baik < 500 else "buruk"
        print(
            f"\nPERHATIAN: data kelas '{kurang}' saat ini kurang dari 500 -- "
            f"kalau target skenariomu 500/500, kamu perlu scraping tambahan "
            f"(khususnya cari lebih banyak review rating rendah, karena review "
            f"negatif biasanya lebih jarang muncul secara natural)."
        )
    else:
        print("\nCukup untuk diambil sampel 500/500 secara acak nanti di tahap splitting.")
    
if __name__ == "__main__":
    main()