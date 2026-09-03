import re
import html
import pandas as pd


INPUT_FILE = "reviews_tokopedia.csv"
OUTPUT_FILE = "reviews_tokopedia_clean.csv"


# Daftar frasa template/otomatis yang umum dipakai e-commerce ketika pembeli
# tidak menulis komentar asli. PENTING: ini daftar awal berdasarkan pola
# umum -- setelah kamu lihat data ASLI hasil scraping Tokopedia, kemungkinan
# perlu ditambah/disesuaikan (pola default Tokopedia bisa beda dari Shopee,
# kita belum pernah lihat contoh aslinya).
FRASA_TEMPLATE = [
    "barang sudah sampai",
    "barang telah sampai",
    "produk sudah sampai",
    "barang diterima dengan baik",
    "barang sudah diterima",
    "sesuai deskripsi",       # kemungkinan pola umum Tokopedia, perlu dicek
    "terima kasih",           # sering muncul BERDIRI SENDIRI sebagai review generik
]

def cek_template(teks, frasa_template):
    """Cek apakah teks review termasuk template otomatis (bukan komentar asli).
    Dianggap template kalau setelah dibersihkan spasi/tanda baca, teksnya
    PERSIS SAMA dengan salah satu frasa template (bukan cuma 'mengandung')
    -- supaya review asli yang KEBETULAN menyebut frasa itu di tengah
    kalimat panjang tidak ikut kebuang."""
    teks_bersih = re.sub(r"[^\w\s]", "", str(teks).lower()).strip()
    return teks_bersih in frasa_template

def main():
    # Baca data dan cek data awal
    df = pd.read_csv(INPUT_FILE)
    total_awal = len(df)
    print(f"Data awal: {total_awal} baris")
    
    #Melihat jumlah entity yang ada karakter dari HTML  
    jumlah_entity = df["text"].str.contains("&amp;|&lt;|&gt;|&#", regex=True).sum()
    
    #Menghapus karakter tadi dengan fungsi unescape dari lib HTML
    df["text"] = df["text"].apply(html.unescape)
    print(f"HTML entity di-decode di {jumlah_entity} baris (misal '&amp;' -> '&')")
    
    # ===== 1. Hapus duplikat (teks sama persis) =====
    # dengan fungsi dari pandas menghapus kolom text yang ada duplikatnya
    df = df.drop_duplicates(subset="text")
    setelah_dedup = len(df)
    print(f"Setelah hapus duplikat: {setelah_dedup} baris "
          f"({total_awal - setelah_dedup} dibuang)")

     # ===== 2. Hapus review template/otomatis =====
    mask_template = df["text"].apply(lambda t: cek_template(t, FRASA_TEMPLATE))
    print(f"\nDitemukan {mask_template.sum()} review template/otomatis:")
    if mask_template.sum() > 0:
        print(df[mask_template]["text"].value_counts().head(10))
 
    df = df[~mask_template]
    setelah_template = len(df)
    print(f"\nSetelah hapus template: {setelah_template} baris "
          f"({setelah_dedup - setelah_template} dibuang)")
 
    # ===== Ringkasan per brand =====
    print("\n" + "=" * 50)
    print("SISA DATA PER BRAND (setelah cleaning)")
    print("=" * 50)
    print(df["brand"].value_counts())
 
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nData bersih tersimpan di '{OUTPUT_FILE}'")
    print(f"Total dibuang: {total_awal - setelah_template} dari {total_awal} baris "
          f"({(total_awal - setelah_template) / total_awal * 100:.1f}%)")
 

if __name__ == "__main__":
    main()