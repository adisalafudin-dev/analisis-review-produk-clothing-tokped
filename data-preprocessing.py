import re
import pandas as pd
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

INPUT_FILE = "dataset_labeled.csv"
STOPWORD_FILE = "stopwords_id.txt"
KAMUS_ALAY_FILE = "kamus_alay_indonesia.csv"
OUTPUT_FILE = "dataset_preprocessed.csv"

# 1. Case folding
# Membuat teks menjadi lowercase
def case_folding(teks):
    return str(teks).lower()

# 2. Remove prunctuation
def remove_punctuation(teks):
    """Buang emoji, tanda baca, dan angka. Hanya sisakan huruf dan spasi."""
    teks = re.sub(r"http\S+|www\S+", " ", teks)   # URL
    teks = re.sub(r"[^\x00-\x7F]+", " ", teks)     # buang emoji/karakter non-ASCII
    teks = re.sub(r"[^a-z\s]", " ", teks)          # buang selain huruf
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks

# ----- 3. Word Normalization -----
def muat_kamus_alay():
    """Memuat kamus normalisasi kata alay/singkatan -> bentuk baku."""
    df = pd.read_csv(KAMUS_ALAY_FILE)
    return dict(zip(df["slang"], df["formal"]))

def word_normalization(teks, kamus_alay):
    """Ganti tiap kata alay/singkatan dengan bentuk bakunya kalau ada di kamus."""
    kata_kata = teks.split()
    hasil = [kamus_alay.get(k, k) for k in kata_kata]
    return " ".join(hasil)

# ----- 4. Stopword Removal -----
def muat_stopwords():
    with open(STOPWORD_FILE, encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f if line.strip())
    stopwords |= {
        "nya", "pokoknya", "yg", "dong", "sih", "kok", "deh", "loh", "lho",
        "gitu", "gini", "aja", "banget", "bgt", "dgn", "utk", "trs", "trus",
        "krn", "tp", "tdk", "dr", "sy", "kalo", "pas",
    }
 
    # PENTING: kata negasi HARUS dikecualikan dari stopword removal.
    # Kalau ikut dibuang, kalimat seperti "gak tebel" (artinya: bagus, pas)
    # bisa jadi cuma "tebal" setelah preprocessing -- makna sentimennya
    # hilang atau bahkan terbalik. Ini bug nyata yang ketemu waktu ngetes
    # data asli (kata "gak" dinormalisasi jadi "enggak", lalu "enggak"
    # ternyata ada di daftar stopword Tala -- dua proses saling bertabrakan).
    kata_negasi = {"tidak", "enggak", "nggak", "gak", "bukan", "jangan", "belum", "tanpa"}
    stopwords -= kata_negasi
 
    return stopwords

def stopword_removal(tokens, stopwords):
    return [t for t in tokens if t not in stopwords and len(t) > 1]

# ----- 5. Stemming -----
def buat_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()
 
 
def stemming(tokens, stemmer):
    return [stemmer.stem(t) for t in tokens]

def preprocess_pipeline(teks, kamus_alay, stopwords, stemmer):
    """Menjalankan seluruh pipeline preprocessing sesuai urutan roadmap."""
    teks = case_folding(teks)                        # 1
    teks = remove_punctuation(teks)                   # 2
    teks = word_normalization(teks, kamus_alay)        # 3
    tokens = word_tokenize(teks)                      # 6 (tokenisasi dulu sebelum stopword/stem)
    tokens = stopword_removal(tokens, stopwords)       # 4
    tokens = stemming(tokens, stemmer)                 # 5
    return " ".join(tokens)
 
 
def main():
    print("Memuat data...")
    df = pd.read_csv(INPUT_FILE)
 
    print("Memuat kamus alay, stopword, & stemmer...")
    kamus_alay = muat_kamus_alay()
    stopwords = muat_stopwords()
    stemmer = buat_stemmer()
    print(f"  Kamus alay : {len(kamus_alay)} kata")
    print(f"  Stopword   : {len(stopwords)} kata")
 
    print(f"\nMemproses {len(df)} review (bisa agak lama karena stemming)...")
    hasil = []
    for i, teks in enumerate(df["text"], 1):
        hasil.append(preprocess_pipeline(teks, kamus_alay, stopwords, stemmer))
        if i % 50 == 0:
            print(f"  {i}/{len(df)} selesai...")
 
    df["text_clean"] = hasil
 
    sebelum = len(df)
    df = df[df["text_clean"].str.strip() != ""]
    sesudah = len(df)
    if sebelum != sesudah:
        print(f"\n{sebelum - sesudah} baris dibuang karena hasil preprocessing kosong.")
 
    df_final = df[["text", "text_clean", "status", "brand"]]
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
 
    print(f"\nSelesai! Tersimpan di '{OUTPUT_FILE}'")
    print("\nContoh perbandingan sebelum vs sesudah:")
    for i in range(3):
        print(f"\n[{i}] SEBELUM: {df_final.iloc[i]['text'][:100]}")
        print(f"    SESUDAH: {df_final.iloc[i]['text_clean']}")
 
    # Contoh yang menampilkan efek word normalization (kata alay -> baku)
    print("\n" + "=" * 50)
    print("CONTOH EFEK WORD NORMALIZATION")
    print("=" * 50)
    contoh_alay = ["bgt", "gk", "yg", "dgn", "udh", "sm", "jd", "krn"]
    for kata in contoh_alay:
        if kata in kamus_alay:
            print(f"  '{kata}' -> '{kamus_alay[kata]}'")
 
 
if __name__ == "__main__":
    main()
 

