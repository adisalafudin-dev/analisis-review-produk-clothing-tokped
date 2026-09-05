import nltk
# Unduh library yang diperlukan saat pertama kali menjalankan
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
 
# ============ KONFIGURASI ============
TRAIN_FILE = "data_train.csv"
TEST_FILE = "data_test.csv"
OUTPUT_FILE = "fitur_semua_metode.pkl"
 
W2V_VECTOR_SIZE = 100   # dimensi vektor Word2Vec/Doc2Vec
W2V_WINDOW = 5          # jarak konteks kata
W2V_MIN_COUNT = 1       # minimal kemunculan kata (dikecilkan karena corpus kecil)
W2V_EPOCHS = 40         # dinaikkan karena corpus kecil (butuh lebih banyak epoch)

def tokenisasi_sederhana(teks):
    """Pecah teks yang sudah dipreprocessing jadi list kata (sudah bersih
    dari tanda baca sejak Tahap 3, jadi cukup split spasi)."""
    return word_tokenize(teks)


def func_bow(train_texts, test_texts):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    print(f"  BOW      : {X_train.shape[1]} fitur (kata unik)")
    return X_train, X_test, vectorizer


def func_tfidf(train_texts, test_texts):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    print(f"  BOW      : {X_train.shape[1]} fitur (kata unik)")
    return X_train, X_test, vectorizer


def func_word2vec(train_tokens, test_tokens):
    model = Word2Vec(
        sentences=train_tokens,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        epochs=W2V_EPOCHS,
        workers=1,
        seed=42
    )
    
    def dokumen_ke_vektor(tokens):
        vektor_kata = [model.wv[t] for t in tokens if t in model.wv]
        if not vektor_kata:
            return np.zeros(W2V_VECTOR_SIZE)
        return np.mean(vektor_kata, axis=0)

    X_train = np.array([dokumen_ke_vektor(t) for t in train_tokens])
    X_test = np.array([dokumen_ke_vektor(t) for t in test_tokens])
 
    kosakata = len(model.wv)
    print(f"  Word2Vec : {kosakata} kata dalam vocabulary, {W2V_VECTOR_SIZE} dimensi/dokumen")
    return X_train, X_test, model

# ===== 4. Doc2Vec =====
def func_doc2vec(train_tokens, test_tokens):
    tagged_train = [TaggedDocument(words=t, tags=[str(i)]) for i, t in enumerate(train_tokens)]
 
    model = Doc2Vec(
        documents=tagged_train,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        epochs=W2V_EPOCHS,
        workers=1,
        seed=42,
    )
 
    X_train = np.array([model.dv[str(i)] for i in range(len(train_tokens))])
    # Untuk data test, dokumennya BELUM pernah dilihat model -> pakai infer_vector
    X_test = np.array([model.infer_vector(t) for t in test_tokens])
 
    print(f"  Doc2Vec  : {W2V_VECTOR_SIZE} dimensi/dokumen")
    return X_train, X_test, model


def main():
    print("Memuat data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    train_df["text_clean"] = train_df["text_clean"].fillna("")
    test_df["text_clean"] = test_df["text_clean"].fillna("")
 
    train_texts = train_df["text_clean"].tolist()
    test_texts = test_df["text_clean"].tolist()
    train_tokens = [tokenisasi_sederhana(t) for t in train_texts]
    test_tokens = [tokenisasi_sederhana(t) for t in test_texts]
 
    y_train = train_df["status"].values
    y_test = test_df["status"].values
 
    print(f"Data training: {len(train_texts)}, testing: {len(test_texts)}")
    print("\nMembuat fitur dengan 4 metode...")
 
    X_train_bow, X_test_bow, bow_vec = func_bow(train_texts, test_texts)
    X_train_tfidf, X_test_tfidf, tfidf_vec = func_tfidf(train_texts, test_texts)
    X_train_w2v, X_test_w2v, w2v_model = func_word2vec(train_tokens, test_tokens)
    X_train_d2v, X_test_d2v, d2v_model = func_doc2vec(train_tokens, test_tokens)
 
    hasil = {
        "y_train": y_train,
        "y_test": y_test,
        "bow": {"X_train": X_train_bow, "X_test": X_test_bow},
        "tfidf": {"X_train": X_train_tfidf, "X_test": X_test_tfidf},
        "word2vec": {"X_train": X_train_w2v, "X_test": X_test_w2v},
        "doc2vec": {"X_train": X_train_d2v, "X_test": X_test_d2v},
    }
 
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(hasil, f)
 
    # Simpan juga model/vectorizer masing-masing (buat inspeksi/pemakaian ulang)
    with open("bow_vectorizer.pkl", "wb") as f:
        pickle.dump(bow_vec, f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vec, f)
    w2v_model.save("word2vec_model.bin")
    d2v_model.save("doc2vec_model.bin")
 
    print(f"\nSemua fitur tersimpan di '{OUTPUT_FILE}'")
    print("\nRingkasan ukuran fitur:")
    print(f"  BOW      : {X_train_bow.shape}")
    print(f"  TF-IDF   : {X_train_tfidf.shape}")
    print(f"  Word2Vec : {X_train_w2v.shape}")
    print(f"  Doc2Vec  : {X_train_d2v.shape}")
 
    print("\nSiap lanjut ke Tahap 6: Modeling XGBoost (4x, satu per metode fitur).")
 
 
if __name__ == "__main__":
    main()