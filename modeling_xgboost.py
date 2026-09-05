import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

# ============ KONFIGURASI ============ 
# Didapat dari data extraction
INPUT_FILE = "fitur_semua_metode.pkl"
OUTPUT_FILE = "hasil_model_xgboost.pkl"
RANDOM_STATE = 42
# ======================================

METODE = ["bow", "tfidf", "word2vec", "doc2vec"]

def encode_label(y):
    return np.array([1 if label == "baik" else 0 for label in y])

def hitung_scale_pos_weight(y_train_encoded):
    """scale_pos_weight = jumlah kelas mayoritas / jumlah kelas minoritas,
    dihitung dari data TRAINING saja."""
    jumlah_positif = (y_train_encoded == 1).sum()  # baik (mayoritas)
    jumlah_negatif = (y_train_encoded == 0).sum()  # buruk (minoritas)
    return jumlah_positif / jumlah_negatif

def main():
    print("Memuat fitur dari Tahap 5...")
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)
    
    y_train = encode_label(data["y_train"])
    y_test = encode_label(data["y_test"])

    spw = hitung_scale_pos_weight(y_train)
    print(f"\nDistribusi label training -- baik: {(y_train==1).sum()}, buruk: {(y_train==0).sum()}")
    print(f"scale_pos_weight yang dipakai: {spw:.2f}")
    print("(artinya: tiap 1 kesalahan di kelas 'buruk' dianggap ~{:.0f}x lebih 'mahal' "
          "dibanding kesalahan di kelas 'baik' saat training)".format(spw))
    
    hasil_semua_metode = {"y_test": y_test}
    
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST UNTUK TIAP METODE FITUR")
    print("=" * 60)
    
    for metode in METODE:
        print(f"\n--- {metode.upper()} ---")
        X_train = data[metode]["X_train"]
        X_test = data[metode]["X_test"]
 
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
 
        pred = model.predict(X_test)
        pred_train = model.predict(X_train)
 
        akurasi_test = accuracy_score(y_test, pred)
        akurasi_train = accuracy_score(y_train, pred_train)
        f1_macro = f1_score(y_test, pred, average="macro", zero_division=0)
 
        print(f"  Akurasi train : {akurasi_train:.2%}")
        print(f"  Akurasi test  : {akurasi_test:.2%}")
        print(f"  F1-macro test : {f1_macro:.3f}  <- metrik utama (bukan akurasi, karena imbalanced)")
 
        hasil_semua_metode[metode] = {
            "model": model,
            "y_pred": pred,
            "akurasi_test": akurasi_test,
            "f1_macro_test": f1_macro,
        }
 
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(hasil_semua_metode, f)
 
    print(f"\nSemua model & hasil prediksi tersimpan di '{OUTPUT_FILE}'")
 
    # Preview perbandingan cepat (evaluasi lengkap di Tahap 7)
    print("\n" + "=" * 60)
    print("PREVIEW PERBANDINGAN (F1-macro test) -- evaluasi lengkap di Tahap 7")
    print("=" * 60)
    urutan = sorted(METODE, key=lambda m: hasil_semua_metode[m]["f1_macro_test"], reverse=True)
    for i, m in enumerate(urutan, 1):
        print(f"  {i}. {m:10s} F1-macro: {hasil_semua_metode[m]['f1_macro_test']:.3f}")
 
 
 
if __name__ == "__main__":
    main()