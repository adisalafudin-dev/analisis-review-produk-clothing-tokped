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

