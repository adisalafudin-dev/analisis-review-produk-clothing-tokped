"""
Word Cloud per Kelas (baik/buruk)
======================================
Visualisasi kata yang paling sering muncul di tiap kelas sentimen.

Cara pakai:
    python wordcloud_ecommerce.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ============ KONFIGURASI ============
INPUT_FILE = "dataset_preprocessed.csv"
OUTPUT_FILE = "wordcloud_baik_buruk.png"
# ======================================


def main():
    df = pd.read_csv(INPUT_FILE)
    df["text_clean"] = df["text_clean"].fillna("")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    warna = {"baik": "Greens", "buruk": "Reds"}

    for i, status in enumerate(["baik", "buruk"]):
        teks_gabungan = " ".join(df[df["status"] == status]["text_clean"])
        n = (df["status"] == status).sum()

        wc = WordCloud(
            width=700, height=500, background_color="white",
            colormap=warna[status], collocations=False,
        ).generate(teks_gabungan)

        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Status: {status} (n={n})", fontsize=14)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    print(f"Word cloud tersimpan di '{OUTPUT_FILE}'")

    # Kata paling sering muncul per kelas (versi angka, pelengkap visual)
    print("\nTop 15 kata paling sering per kelas:")
    for status in ["baik", "buruk"]:
        print(f"\n--- {status} ---")
        semua_kata = " ".join(df[df["status"] == status]["text_clean"]).split()
        top = pd.Series(semua_kata).value_counts().head(15)
        print(top.to_string())


if __name__ == "__main__":
    main()