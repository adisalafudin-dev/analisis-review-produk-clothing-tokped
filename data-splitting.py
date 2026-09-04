import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "dataset_preprocessed.csv"
OUTPUT_TRAIN = "data_train.csv"
OUTPUT_TEST = "data_test.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Total data: {len(df)} baris")
    print("\nDistribusi status keseluruhan:")
    print(df["status"].value_counts())
 
    jumlah_buruk = (df["status"] == "buruk").sum()
    if jumlah_buruk < 10:
        print(
            f"\nPERHATIAN: cuma {jumlah_buruk} data 'buruk' di seluruh dataset. "
            f"Dengan split 80/20, test set kemungkinan cuma dapat "
            f"~{round(jumlah_buruk * TEST_SIZE)} data buruk -- evaluasi nanti "
            f"di kelas ini kemungkinan besar tidak stabil/kurang bisa "
            f"dipercaya, murni karena keterbatasan jumlah data (bukan salah "
            f"kode). Ini sama persis pelajaran dari studi kasus SVM sebelumnya."
        )
 
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["status"]
    )
 
    print(f"\nData training : {len(train_df)} baris")
    print(train_df["status"].value_counts())
    print(f"\nData testing  : {len(test_df)} baris")
    print(test_df["status"].value_counts())
 
    train_df.to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8-sig")
    test_df.to_csv(OUTPUT_TEST, index=False, encoding="utf-8-sig")
 
    print(f"\nTersimpan: '{OUTPUT_TRAIN}' dan '{OUTPUT_TEST}'")
    print("\nSiap lanjut ke Tahap 5: Ekstraksi Fitur (BOW, TF-IDF, Word2Vec, Doc2Vec).")
 
 
if __name__ == "__main__":
    main()