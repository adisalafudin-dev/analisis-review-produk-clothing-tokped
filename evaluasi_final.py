import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from xgboost import XGBClassifier

# ============ KONFIGURASI ============
HASIL_MODEL_FILE = "hasil_model_xgboost.pkl"
FULL_DATA_FILE = "dataset_preprocessed.csv"  # train+test digabung, untuk cross-validation
OUTPUT_CONFUSION = "evaluasi_confusion_matrix.png"
OUTPUT_PERBANDINGAN = "evaluasi_perbandingan_metode.png"
METODE = ["bow", "tfidf", "word2vec", "doc2vec"]
LABEL_NAMA = ["buruk", "baik"]  # urutan: 0=buruk, 1=baik
CV_FOLDS = 3  # dibatasi 3 karena kelas buruk cuma 9 data total di seluruh dataset
RANDOM_STATE = 42
W2V_VECTOR_SIZE = 100
# ======================================


def encode_label(y):
    return np.array([1 if label == "baik" else 0 for label in y])

def evaluasi_hasil_split(hasil):
    y_test = hasil["y_test"]
 
    print("=" * 60)
    print("1. CLASSIFICATION REPORT PER METODE (dari split 80/20)")
    print("=" * 60)
    for metode in METODE:
        pred = hasil[metode]["y_pred"]
        print(f"\n--- {metode.upper()} ---")
        print(classification_report(y_test, pred, target_names=LABEL_NAMA, zero_division=0))
 
    print("\n" + "=" * 60)
    print("2. CONFUSION MATRIX PER METODE")
    print("=" * 60)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, metode in enumerate(METODE):
        pred = hasil[metode]["y_pred"]
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        im = axes[i].imshow(cm, cmap="Blues")
        axes[i].set_title(metode.upper())
        axes[i].set_xticks([0, 1]); axes[i].set_xticklabels(LABEL_NAMA)
        axes[i].set_yticks([0, 1]); axes[i].set_yticklabels(LABEL_NAMA)
        axes[i].set_xlabel("Prediksi"); axes[i].set_ylabel("Label Asli")
        for r in range(2):
            for c in range(2):
                axes[i].text(c, r, str(cm[r, c]), ha="center", va="center",
                             color="white" if cm[r, c] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(OUTPUT_CONFUSION, dpi=150)
    print(f"Confusion matrix tersimpan di '{OUTPUT_CONFUSION}'")
 
 
# ===== 3: Cross-validation di seluruh dataset =====
def buat_fitur_untuk_fold(train_texts, test_texts, train_tokens, test_tokens):
    """Bikin ulang keempat jenis fitur, khusus untuk 1 fold CV (fit cuma dari
    train fold, transform ke test fold -- sama prinsipnya kayak Tahap 5)."""
    hasil = {}
 
    bow_vec = CountVectorizer()
    hasil["bow"] = (bow_vec.fit_transform(train_texts), bow_vec.transform(test_texts))
 
    tfidf_vec = TfidfVectorizer()
    hasil["tfidf"] = (tfidf_vec.fit_transform(train_texts), tfidf_vec.transform(test_texts))
 
    w2v = Word2Vec(sentences=train_tokens, vector_size=W2V_VECTOR_SIZE, window=5,
                   min_count=1, epochs=40, workers=1, seed=RANDOM_STATE)
    def doc_vec_w2v(tokens):
        v = [w2v.wv[t] for t in tokens if t in w2v.wv]
        return np.mean(v, axis=0) if v else np.zeros(W2V_VECTOR_SIZE)
    hasil["word2vec"] = (
        np.array([doc_vec_w2v(t) for t in train_tokens]),
        np.array([doc_vec_w2v(t) for t in test_tokens]),
    )
 
    tagged = [TaggedDocument(words=t, tags=[str(i)]) for i, t in enumerate(train_tokens)]
    d2v = Doc2Vec(documents=tagged, vector_size=W2V_VECTOR_SIZE, window=5,
                  min_count=1, epochs=40, workers=1, seed=RANDOM_STATE)
    hasil["doc2vec"] = (
        np.array([d2v.dv[str(i)] for i in range(len(train_tokens))]),
        np.array([d2v.infer_vector(t) for t in test_tokens]),
    )
 
    return hasil
 
 
def cross_validation_semua_metode(df):
    print("\n" + "=" * 60)
    print(f"3. CROSS VALIDATION ({CV_FOLDS}-FOLD) DI SELURUH DATASET")
    print("=" * 60)
    print("(Menggunakan kesembilan data buruk secara bergantian sebagai data uji,")
    print(" jauh lebih informatif daripada cuma 2 data buruk di 1x split test)\n")
 
    texts = df["text_clean"].fillna("").tolist()
    tokens = [str(t).split() for t in texts]
    y = encode_label(df["status"].values)
 
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    skor_per_metode = {m: [] for m in METODE}
 
    for fold_ke, (idx_train, idx_test) in enumerate(skf.split(texts, y), 1):
        print(f"Fold {fold_ke}/{CV_FOLDS}...")
        train_texts = [texts[i] for i in idx_train]
        test_texts = [texts[i] for i in idx_test]
        train_tokens = [tokens[i] for i in idx_train]
        test_tokens = [tokens[i] for i in idx_test]
        y_train_fold, y_test_fold = y[idx_train], y[idx_test]
 
        spw = (y_train_fold == 1).sum() / (y_train_fold == 0).sum()
        fitur = buat_fitur_untuk_fold(train_texts, test_texts, train_tokens, test_tokens)
 
        for metode in METODE:
            X_train, X_test = fitur[metode]
            model = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                scale_pos_weight=spw, eval_metric="logloss", random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train_fold)
            pred = model.predict(X_test)
            f1 = f1_score(y_test_fold, pred, average="macro", zero_division=0)
            skor_per_metode[metode].append(f1)
 
    print("\nHasil F1-macro per fold:")
    for metode in METODE:
        skor = skor_per_metode[metode]
        print(f"  {metode:10s}: {[round(s,3) for s in skor]}  (rata-rata: {np.mean(skor):.3f})")
 
    return skor_per_metode
 
 
def main():
    with open(HASIL_MODEL_FILE, "rb") as f:
        hasil = pickle.load(f)
 
    evaluasi_hasil_split(hasil)
 
    df_full = pd.read_csv(FULL_DATA_FILE)
    skor_cv = cross_validation_semua_metode(df_full)
 
    # ===== Grafik perbandingan =====
    rata_rata = {m: np.mean(skor_cv[m]) for m in METODE}
    fig, ax = plt.subplots(figsize=(8, 5))
    warna = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = ax.bar(rata_rata.keys(), rata_rata.values(), color=warna)
    ax.set_ylabel("F1-macro (rata-rata cross-validation)")
    ax.set_title("Perbandingan 4 Metode Ekstraksi Fitur (XGBoost)")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, rata_rata.values()):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(OUTPUT_PERBANDINGAN, dpi=150)
    print(f"\nGrafik perbandingan tersimpan di '{OUTPUT_PERBANDINGAN}'")
 
    # ===== Kesimpulan =====
    print("\n" + "=" * 60)
    print("4. KESIMPULAN AKHIR")
    print("=" * 60)
    urutan = sorted(rata_rata.items(), key=lambda x: x[1], reverse=True)
    for i, (m, skor) in enumerate(urutan, 1):
        print(f"  {i}. {m:10s} F1-macro rata-rata: {skor:.3f}")
 
    terbaik, skor_terbaik = urutan[0]
    print(f"\nBerdasarkan cross-validation, metode terbaik: '{terbaik}' "
          f"(F1-macro: {skor_terbaik:.3f})")
    print(
        "\nCATATAN PENTING: dengan cuma 9 data 'buruk' di SELURUH dataset, hasil ini "
        "tetap punya ketidakpastian statistik yang besar -- perbedaan F1-macro antar "
        "metode di sini sebagian bisa jadi cuma kebetulan (noise), bukan murni karena "
        "satu metode memang lebih unggul. Kesimpulan yang lebih kuat butuh data buruk "
        "yang jauh lebih banyak. Ini sendiri adalah temuan valid untuk bagian "
        "keterbatasan penelitian di laporanmu."
    )
 
 
if __name__ == "__main__":
    main()