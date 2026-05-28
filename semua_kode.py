"""
=============================================================================
SEMUA KODE MINI SEARCH ENGINE - GABUNGAN DALAM SATU FILE
=============================================================================

File ini menggabungkan 3 file Python menjadi satu agar mudah dipelajari:
  1. mini_search_engine.py  → Bagian 1: Search Engine CLI (Command Line Interface)
  2. generate_report.py     → Bagian 2: Generate Laporan PDF
  3. app.py                 → Bagian 3: Web UI dengan Streamlit

Cara menjalankan:
  - Untuk CLI search engine  : python semua_kode.py --mode cli
  - Untuk generate PDF       : python semua_kode.py --mode pdf
  - Untuk web UI Streamlit   : streamlit run semua_kode.py -- --mode web
  - Tanpa argumen            : Akan muncul pilihan menu

Dependensi:
  pip install Sastrawi fpdf2 streamlit pandas

Mata Kuliah: Temu Kembali Informasi - UTS
=============================================================================
"""

import csv
import re
import math
import os
import sys
import io
import argparse
from collections import defaultdict

# Fix encoding untuk Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stdin.encoding != 'utf-8':
    try:
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


# ############################################################################
#
#   BAGIAN 0: SETUP SASTRAWI (DIPAKAI OLEH SEMUA BAGIAN)
#
# ############################################################################

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    print("=" * 60)
    print("  PERINGATAN: Library Sastrawi belum terinstall!")
    print("  Jalankan: pip install Sastrawi")
    print("  Program akan tetap berjalan tanpa stemming.")
    print("=" * 60)
    print()

# Daftar stopword sederhana sebagai fallback jika Sastrawi tidak tersedia
FALLBACK_STOP_WORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan',
    'untuk', 'pada', 'adalah', 'dalam', 'tidak', 'akan', 'sudah',
    'juga', 'saya', 'aku', 'kamu', 'dia', 'kami', 'kita', 'mereka',
    'ada', 'bisa', 'atau', 'ya', 'nya', 'se', 'tapi', 'karena',
    'kalau', 'lagi', 'mau', 'apa', 'sama', 'kan', 'aja', 'sih',
    'dong', 'deh', 'loh', 'kok', 'jadi', 'udah', 'gak', 'ga',
    'gk', 'yg', 'tp', 'tdk', 'sm', 'krn', 'krna', 'klo', 'dg',
    'the', 'a', 'an', 'of', 'to', 'in', 'is', 'it', 'for'
}

# Path file CSV (dipakai oleh semua bagian)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "tokenisasi dan stopwatch removal.csv")


# ############################################################################
#
#   BAGIAN 1: KELAS MINI SEARCH ENGINE
#   (Asal file: mini_search_engine.py & app.py)
#
#   Kelas ini berisi semua logika inti:
#   - Pre-processing (case folding, punctuation removal, stopword, stemming)
#   - Inverted Index
#   - TF-IDF Weighting (Log Frequency)
#   - Vector Space Model dengan Cosine Similarity
#
# ############################################################################

class MiniSearchEngine:
    """
    Mesin Pencari Mini yang mengimplementasikan:
    - Text Pre-processing (case folding, punctuation removal, stopword removal, stemming)
    - Inverted Index
    - TF-IDF Weighting (Log Frequency)
    - Vector Space Model dengan Cosine Similarity
    """

    def __init__(self):
        # Inisialisasi Sastrawi Stemmer dan StopWord Remover
        if SASTRAWI_AVAILABLE:
            stemmer_factory = StemmerFactory()
            self.stemmer = stemmer_factory.create_stemmer()

            stopword_factory = StopWordRemoverFactory()
            self.stop_words = set(stopword_factory.get_stop_words())
        else:
            self.stemmer = None
            self.stop_words = FALLBACK_STOP_WORDS.copy()

        # Data structures
        self.documents = {}           # {doc_id: original_text}
        self.doc_sources = {}         # {doc_id: source}
        self.processed_docs = {}      # {doc_id: [term1, term2, ...]}
        self.inverted_index = {}      # {term: {doc_id: tf, ...}}
        self.tfidf_weights = {}       # {term: {doc_id: tfidf_weight, ...}}
        self.doc_lengths = {}         # {doc_id: vector_length} untuk normalisasi cosine
        self.num_docs = 0
        self.vocabulary = set()

    # ========================================================================
    # 1. PRE-PROCESSING
    # ========================================================================

    def case_folding(self, text):
        """Mengubah semua teks menjadi huruf kecil (lowercase)."""
        return text.lower()

    def remove_punctuation(self, text):
        """Menghapus tanda baca, angka, dan karakter khusus."""
        # Hapus URL
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Hapus emoji dan karakter unicode khusus
        text = re.sub(r'[^\w\s]', ' ', text)
        # Hapus angka
        text = re.sub(r'\d+', '', text)
        # Hapus spasi berlebihan
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_stopwords(self, tokens):
        """Menghapus stop words dari daftar token."""
        return [token for token in tokens if token not in self.stop_words and len(token) > 1]

    def stem_tokens(self, tokens):
        """Melakukan stemming pada setiap token menggunakan Sastrawi."""
        if self.stemmer:
            return [self.stemmer.stem(token) for token in tokens]
        return tokens  # Kembalikan tanpa stemming jika Sastrawi tidak tersedia

    def preprocess(self, text):
        """
        Pipeline pre-processing lengkap:
        1. Case folding
        2. Punctuation removal
        3. Tokenization
        4. Stopword removal
        5. Stemming
        """
        # Step 1: Case folding
        text = self.case_folding(text)

        # Step 2: Punctuation removal
        text = self.remove_punctuation(text)

        # Step 3: Tokenization (split menjadi kata-kata)
        tokens = text.split()

        # Step 4: Stopword removal
        tokens = self.remove_stopwords(tokens)

        # Step 5: Stemming
        tokens = self.stem_tokens(tokens)

        return tokens

    def preprocess_steps(self, text):
        """Mengembalikan hasil setiap langkah pre-processing (untuk UI)."""
        steps = {}
        steps['original'] = text
        step1 = self.case_folding(text)
        steps['case_folding'] = step1
        step2 = self.remove_punctuation(step1)
        steps['punctuation_removal'] = step2
        tokens = step2.split()
        steps['tokenization'] = tokens.copy()
        after_sw = self.remove_stopwords(tokens)
        steps['stopword_removal'] = after_sw.copy()
        after_stem = self.stem_tokens(after_sw)
        steps['stemming'] = after_stem.copy()
        return steps

    # ========================================================================
    # 2. INDEXING - Membangun Inverted Index
    # ========================================================================

    def load_documents_from_csv(self, filepath):
        """
        Memuat dokumen dari file CSV.
        Membaca kolom 'Komentar' sebagai dokumen.
        """
        print(f"\n{'='*60}")
        print(f"  MEMUAT DOKUMEN DARI: {os.path.basename(filepath)}")
        print(f"{'='*60}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                reader = csv.reader(f)
                rows = list(reader)

        # Cari baris header (exact match untuk nama kolom)
        header_row = None
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                if cell.strip().lower() == 'komentar':
                    header_row = i
                    break
            if header_row is not None:
                break

        if header_row is None:
            print("  [ERROR] Kolom 'Komentar' tidak ditemukan di CSV!")
            return 0

        # Cari indeks kolom
        header = rows[header_row]
        komentar_idx = None
        no_idx = None
        sumber_idx = None

        for j, cell in enumerate(header):
            cell_lower = cell.strip().lower()
            if cell_lower == 'komentar':
                komentar_idx = j
            elif cell_lower == 'no':
                no_idx = j
            elif cell_lower == 'sumber':
                sumber_idx = j

        if komentar_idx is None:
            print("  [ERROR] Kolom 'Komentar' tidak ditemukan!")
            return 0

        # Baca data dokumen, menangani komentar multi-baris
        doc_count = 0
        current_doc_id = None
        current_text = None
        current_source = None

        for i in range(header_row + 1, len(rows)):
            row = rows[i]
            if len(row) <= komentar_idx:
                # Baris lanjutan dari komentar multi-baris
                if current_doc_id is not None and len(row) > 0:
                    extra_text = ' '.join([cell for cell in row if cell.strip()])
                    if extra_text:
                        current_text += ' ' + extra_text
                continue

            # Cek apakah ini baris dokumen baru (ada nomor)
            no_val = row[no_idx].strip() if no_idx is not None and no_idx < len(row) else ''
            komentar_val = row[komentar_idx].strip() if komentar_idx < len(row) else ''

            if no_val and komentar_val:
                # Simpan dokumen sebelumnya
                if current_doc_id is not None and current_text:
                    self.documents[current_doc_id] = current_text
                    self.doc_sources[current_doc_id] = current_source
                    doc_count += 1

                # Mulai dokumen baru
                try:
                    current_doc_id = int(no_val)
                except ValueError:
                    current_doc_id = doc_count + 1
                current_text = komentar_val
                current_source = row[sumber_idx].strip() if sumber_idx is not None and sumber_idx < len(row) else '-'
            elif current_doc_id is not None and komentar_val:
                # Lanjutan komentar multi-baris
                current_text += ' ' + komentar_val

        # Simpan dokumen terakhir
        if current_doc_id is not None and current_text:
            self.documents[current_doc_id] = current_text
            self.doc_sources[current_doc_id] = current_source
            doc_count += 1

        self.num_docs = len(self.documents)
        print(f"  [OK] Berhasil memuat {self.num_docs} dokumen")
        return self.num_docs

    def build_index(self):
        """
        Membangun Inverted Index dari koleksi dokumen.
        Setiap term dipetakan ke dokumen-dokumen yang mengandungnya beserta frekuensinya.
        """
        print(f"\n{'='*60}")
        print(f"  MEMBANGUN INVERTED INDEX")
        print(f"{'='*60}")
        print(f"  Melakukan pre-processing pada {self.num_docs} dokumen...")

        for doc_id, text in self.documents.items():
            # Pre-process dokumen
            tokens = self.preprocess(text)
            self.processed_docs[doc_id] = tokens

            # Hitung term frequency untuk dokumen ini
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            # Masukkan ke inverted index
            for term, freq in term_freq.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_id] = freq
                self.vocabulary.add(term)

        print(f"  [OK] Jumlah term unik (vocabulary): {len(self.vocabulary)}")
        print(f"  [OK] Inverted index berhasil dibangun")

    # ========================================================================
    # 3. TF-IDF WEIGHTING
    # ========================================================================

    def compute_tfidf(self):
        """
        Menghitung bobot TF-IDF untuk setiap term di setiap dokumen.

        TF (Log Frequency Weighting): 
            w(t,d) = 1 + log10(tf(t,d))  jika tf > 0
            w(t,d) = 0                    jika tf = 0

        IDF (Inverse Document Frequency):
            idf(t) = log10(N / df(t))
            dimana N = jumlah total dokumen, df(t) = jumlah dokumen yang mengandung term t

        TF-IDF:
            w(t,d) = tf_weight * idf
        """
        print(f"\n{'='*60}")
        print(f"  MENGHITUNG BOBOT TF-IDF")
        print(f"{'='*60}")

        N = self.num_docs

        for term in self.vocabulary:
            self.tfidf_weights[term] = {}
            # Document frequency
            df = len(self.inverted_index[term])
            # IDF
            idf = math.log10(N / df) if df > 0 else 0

            for doc_id, tf in self.inverted_index[term].items():
                # Log frequency weighting: 1 + log10(tf)
                tf_weight = 1 + math.log10(tf) if tf > 0 else 0
                # TF-IDF
                tfidf = tf_weight * idf
                self.tfidf_weights[term][doc_id] = tfidf

        # Hitung panjang vektor dokumen (untuk normalisasi cosine similarity)
        for doc_id in self.documents:
            length_sq = 0
            for term in self.vocabulary:
                if term in self.tfidf_weights and doc_id in self.tfidf_weights[term]:
                    weight = self.tfidf_weights[term][doc_id]
                    length_sq += weight ** 2
            self.doc_lengths[doc_id] = math.sqrt(length_sq) if length_sq > 0 else 0

        print(f"  [OK] Bobot TF-IDF berhasil dihitung untuk {len(self.vocabulary)} term")

    # ========================================================================
    # 4. SEARCH (VECTOR SPACE MODEL + COSINE SIMILARITY)
    # ========================================================================

    def search(self, query, top_k=10, normalize=True):
        """
        Melakukan pencarian menggunakan Vector Space Model.
        Menghitung Cosine Similarity antara vektor query dan vektor dokumen.

        Cosine Similarity:
            cos(q, d) = (q · d) / (|q| × |d|)

        Parameter:
            query     : string query pencarian
            top_k     : jumlah hasil teratas yang dikembalikan
            normalize : True = cosine similarity, False = dot product saja

        Returns:
            List of dict {doc_id, score, text, source, token_count}
            List of query_tokens
        """
        # Pre-process query
        query_tokens = self.preprocess(query)

        if not query_tokens:
            return [], query_tokens

        # Hitung TF-IDF untuk query
        query_tf = defaultdict(int)
        for token in query_tokens:
            query_tf[token] += 1

        query_weights = {}
        query_length_sq = 0
        N = self.num_docs

        for term, tf in query_tf.items():
            if term in self.inverted_index:
                df = len(self.inverted_index[term])
                idf = math.log10(N / df) if df > 0 else 0
                tf_weight = 1 + math.log10(tf) if tf > 0 else 0
                tfidf = tf_weight * idf
                query_weights[term] = tfidf
                query_length_sq += tfidf ** 2

        query_length = math.sqrt(query_length_sq) if query_length_sq > 0 else 0

        if query_length == 0:
            return [], query_tokens

        # Hitung cosine similarity untuk setiap dokumen
        scores = {}
        for term, q_weight in query_weights.items():
            if term in self.tfidf_weights:
                for doc_id, d_weight in self.tfidf_weights[term].items():
                    if doc_id not in scores:
                        scores[doc_id] = 0
                    # Dot product (q · d)
                    scores[doc_id] += q_weight * d_weight

        # Normalisasi dengan panjang vektor (cosine similarity)
        results = []
        for doc_id, dot_product in scores.items():
            if normalize:
                doc_length = self.doc_lengths[doc_id]
                if doc_length > 0 and query_length > 0:
                    cosine_sim = dot_product / (query_length * doc_length)
                else:
                    cosine_sim = 0
            else:
                # Tanpa normalisasi - hanya dot product
                cosine_sim = dot_product

            results.append({
                'doc_id': doc_id,
                'score': cosine_sim,
                'text': self.documents[doc_id],
                'source': self.doc_sources.get(doc_id, '-'),
                'token_count': len(self.processed_docs.get(doc_id, []))
            })

        # Urutkan berdasarkan skor tertinggi
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_k], query_tokens

    # ========================================================================
    # 5. DISPLAY METHODS (untuk mode CLI)
    # ========================================================================

    def display_inverted_index(self, max_terms=20):
        """Menampilkan sebagian inverted index."""
        print(f"\n{'='*60}")
        print(f"  INVERTED INDEX (menampilkan {max_terms} term pertama)")
        print(f"{'='*60}")
        print(f"  {'Term':<25} {'DF':>5}  {'Posting List (doc_id: tf)'}")
        print(f"  {'-'*25} {'-'*5}  {'-'*30}")

        for i, (term, postings) in enumerate(sorted(self.inverted_index.items())):
            if i >= max_terms:
                remaining = len(self.vocabulary) - max_terms
                print(f"\n  ... dan {remaining} term lainnya")
                break
            df = len(postings)
            posting_str = ', '.join([f"D{did}:{tf}" for did, tf in sorted(postings.items())])
            # Potong jika terlalu panjang
            if len(posting_str) > 50:
                posting_str = posting_str[:50] + "..."
            print(f"  {term:<25} {df:>5}  {posting_str}")

    def display_tfidf_matrix(self, max_terms=15, max_docs=10):
        """Menampilkan sebagian matriks TF-IDF."""
        print(f"\n{'='*60}")
        print(f"  MATRIKS TF-IDF (sebagian)")
        print(f"{'='*60}")

        # Ambil sebagian term dan dokumen
        sorted_terms = sorted(self.vocabulary)[:max_terms]
        sorted_docs = sorted(self.documents.keys())[:max_docs]

        # Header
        header = f"  {'Term':<20}"
        for doc_id in sorted_docs:
            header += f"{'D'+str(doc_id):>8}"
        print(header)
        print(f"  {'-'*20}" + f"{'-'*8}" * len(sorted_docs))

        for term in sorted_terms:
            row = f"  {term:<20}"
            for doc_id in sorted_docs:
                weight = self.tfidf_weights.get(term, {}).get(doc_id, 0)
                row += f"{weight:>8.4f}"
            print(row)

    def display_search_results(self, query, results):
        """Menampilkan hasil pencarian dengan format yang rapi (CLI)."""
        print(f"\n{'='*60}")
        print(f"  HASIL PENCARIAN")
        print(f"{'='*60}")
        print(f"  Query: \"{query}\"")
        print(f"  Jumlah hasil: {len(results)}")
        print(f"{'='*60}")

        if not results:
            print(f"\n  [!] Tidak ada dokumen yang relevan ditemukan.")
            print(f"      Coba gunakan kata kunci lain.\n")
            return

        for rank, r in enumerate(results, 1):
            # Potong teks jika terlalu panjang
            display_text = r['text'][:150] + "..." if len(r['text']) > 150 else r['text']
            source = r.get('source', '-')

            print(f"\n  +-- Rank #{rank}")
            print(f"  |  Dokumen  : D{r['doc_id']}")
            print(f"  |  Skor     : {r['score']:.6f}")
            print(f"  |  Sumber   : {source if source else '-'}")
            print(f"  |  Komentar : {display_text}")
            print(f"  +{'-'*55}")

    def display_preprocessing_example(self, text):
        """Menampilkan contoh proses pre-processing langkah demi langkah."""
        print(f"\n{'='*60}")
        print(f"  CONTOH PRE-PROCESSING")
        print(f"{'='*60}")
        print(f"  Teks asli       : {text[:80]}...")

        step1 = self.case_folding(text)
        print(f"  Case folding    : {step1[:80]}...")

        step2 = self.remove_punctuation(step1)
        print(f"  Remove punct.   : {step2[:80]}...")

        tokens = step2.split()
        print(f"  Tokenisasi      : {tokens[:10]}...")

        step4 = self.remove_stopwords(tokens)
        print(f"  Stopword removal: {step4[:10]}...")

        step5 = self.stem_tokens(step4)
        print(f"  Stemming        : {step5[:10]}...")
        print(f"  Jumlah token    : {len(tokens)} -> {len(step5)} (setelah preprocessing)")


# ############################################################################
#
#   BAGIAN 2: MODE CLI - PENCARIAN INTERAKTIF
#   (Asal file: mini_search_engine.py)
#
#   Menjalankan search engine di terminal/command prompt.
#   Pengguna bisa mengetik query dan melihat hasil pencarian.
#
# ############################################################################

def run_cli():
    """Menjalankan Mini Search Engine dalam mode CLI (Command Line Interface)."""
    # Header program
    print()
    print("+" + "="*58 + "+")
    print("|" + " "*58 + "|")
    print("|" + "MINI SEARCH ENGINE".center(58) + "|")
    print("|" + "Mesin Pencari Mini - Temu Kembali Informasi".center(58) + "|")
    print("|" + " "*58 + "|")
    print("|" + "Pre-processing | TF-IDF | Cosine Similarity".center(58) + "|")
    print("|" + " "*58 + "|")
    print("+" + "="*58 + "+")

    # Inisialisasi search engine
    engine = MiniSearchEngine()

    if not os.path.exists(CSV_FILE):
        print(f"\n  [ERROR] File CSV tidak ditemukan: {CSV_FILE}")
        print(f"  Pastikan file CSV berada di folder yang sama dengan script ini.")
        sys.exit(1)

    # ========== FASE 1: Load dan Pre-processing ==========
    engine.load_documents_from_csv(CSV_FILE)

    # Tampilkan contoh pre-processing
    if engine.documents:
        first_doc = list(engine.documents.values())[0]
        engine.display_preprocessing_example(first_doc)

    # ========== FASE 2: Build Inverted Index ==========
    engine.build_index()
    engine.display_inverted_index(max_terms=20)

    # ========== FASE 3: Compute TF-IDF ==========
    engine.compute_tfidf()
    engine.display_tfidf_matrix(max_terms=10, max_docs=8)

    # ========== FASE 4: Interactive Search ==========
    print(f"\n{'='*60}")
    print(f"  PENCARIAN INTERAKTIF")
    print(f"{'='*60}")
    print(f"  Masukkan query untuk mencari dokumen relevan.")
    print(f"  Ketik 'quit' atau 'exit' untuk keluar.")
    print(f"  Ketik 'index' untuk melihat inverted index.")
    print(f"  Ketik 'tfidf' untuk melihat matriks TF-IDF.")
    print(f"{'='*60}")

    while True:
        try:
            print()
            query = input("  Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Terima kasih! Sampai jumpa.\n")
            break

        if not query:
            continue

        if query.lower() in ('quit', 'exit', 'keluar', 'q'):
            print(f"\n  Terima kasih telah menggunakan Mini Search Engine!\n")
            break

        if query.lower() == 'index':
            engine.display_inverted_index(max_terms=30)
            continue

        if query.lower() == 'tfidf':
            engine.display_tfidf_matrix(max_terms=20, max_docs=10)
            continue

        # Lakukan pencarian
        results, query_tokens = engine.search(query, top_k=10)

        # Tampilkan hasil
        engine.display_search_results(query, results)

        # Tampilkan detail query processing
        print(f"\n  [i] Query setelah pre-processing: {query_tokens}")
        matched_terms = [t for t in query_tokens if t in engine.inverted_index]
        print(f"  [i] Term yang ditemukan di index: {matched_terms}")


# ############################################################################
#
#   BAGIAN 3: MODE PDF - GENERATE LAPORAN PDF
#   (Asal file: generate_report.py)
#
#   Menghasilkan laporan PDF lengkap berisi:
#   a. Analisis Bobot (IDF)
#   b. Analisis Efek Normalisasi (Cosine Normalization)
#   c. Evaluasi Sistem (Precision, Recall, F-Measure)
#
# ############################################################################

def clean_for_pdf(text):
    """Hapus karakter non-latin1 (emoji, dll) agar kompatibel dengan PDF font Helvetica."""
    return text.encode('latin-1', errors='ignore').decode('latin-1')


def run_pdf():
    """Menjalankan Generate Laporan PDF dengan penjelasan lengkap dan manusiawi."""
    try:
        from fpdf import FPDF
    except ImportError:
        print("  [ERROR] Library fpdf2 belum terinstall!")
        print("  Jalankan: pip install fpdf2")
        sys.exit(1)

    print("Inisialisasi engine untuk laporan PDF...")

    # Inisialisasi engine dan load data
    engine = MiniSearchEngine()

    if not os.path.exists(CSV_FILE):
        print(f"\n  [ERROR] File CSV tidak ditemukan: {CSV_FILE}")
        sys.exit(1)

    engine.load_documents_from_csv(CSV_FILE)
    engine.build_index()
    engine.compute_tfidf()

    N = engine.num_docs
    print(f"Total dokumen: {N}")
    print(f"Vocabulary: {len(engine.vocabulary)} term unik")

    # ================================================================
    # Analisis Data
    # ================================================================
    print("Melakukan analisis...")

    # === a. Analisis Bobot IDF ===
    keyword1 = "imunisasi"
    keyword2 = "kejang"

    df1 = len(engine.inverted_index.get(keyword1, {}))
    df2 = len(engine.inverted_index.get(keyword2, {}))
    idf1 = math.log10(N / df1) if df1 > 0 else 0
    idf2 = math.log10(N / df2) if df2 > 0 else 0

    docs_kw1 = sorted(engine.inverted_index.get(keyword1, {}).keys())
    docs_kw2 = sorted(engine.inverted_index.get(keyword2, {}).keys())

    # === b. Analisis Normalisasi ===
    query_norm = "demam setelah imunisasi"
    results_norm, tokens_norm = engine.search(query_norm, top_k=10, normalize=True)
    results_no_norm, _ = engine.search(query_norm, top_k=10, normalize=False)

    # === c. Evaluasi Sistem ===
    query1 = "demam setelah imunisasi"
    results_q1, tokens_q1 = engine.search(query1, top_k=10, normalize=True)
    ground_truth_q1 = {8, 10, 13, 15, 22, 27, 31, 34, 40, 42, 45, 46}

    query2 = "vaksin anak balita"
    results_q2, tokens_q2 = engine.search(query2, top_k=10, normalize=True)
    ground_truth_q2 = {1, 2, 4, 5, 7, 28, 29, 32, 38, 43, 44, 47}

    def calc_metrics(results, ground_truth, k=10):
        retrieved = set(r['doc_id'] for r in results[:k])
        relevant_retrieved = retrieved & ground_truth
        precision = len(relevant_retrieved) / len(retrieved) if len(retrieved) > 0 else 0
        recall = len(relevant_retrieved) / len(ground_truth) if len(ground_truth) > 0 else 0
        f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f_measure, retrieved, relevant_retrieved

    p1, r1, f1, ret1, rr1 = calc_metrics(results_q1, ground_truth_q1)
    p2, r2, f2, ret2, rr2 = calc_metrics(results_q2, ground_truth_q2)

    # === Preprocessing example untuk Pendahuluan ===
    example_doc_id = list(engine.documents.keys())[0]
    example_text = engine.documents[example_doc_id]
    example_steps = engine.preprocess_steps(example_text)

    # ================================================================
    # PDF Report Generation
    # ================================================================
    print("Membuat laporan PDF...")

    class LaporanPDF(FPDF):
        def __init__(self):
            super().__init__()
            self.set_auto_page_break(auto=True, margin=20)

        def header(self):
            self.set_font('Helvetica', 'B', 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, 'Laporan Analisis Mini Search Engine - Temu Kembali Informasi', align='C')
            self.ln(5)
            self.set_draw_color(79, 70, 229)
            self.set_line_width(0.5)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f'Halaman {self.page_no()}/{{nb}}', align='C')

        def judul_bagian(self, text):
            self.set_font('Helvetica', 'B', 14)
            self.set_text_color(79, 70, 229)
            self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(79, 70, 229)
            self.set_line_width(0.8)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(5)

        def sub_judul(self, text):
            self.set_font('Helvetica', 'B', 11)
            self.set_text_color(30, 30, 30)
            self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

        def paragraf(self, text):
            self.set_font('Helvetica', '', 10)
            self.set_text_color(50, 50, 50)
            self.multi_cell(0, 5.5, text)
            self.ln(2)

        def rumus(self, text):
            self.set_font('Courier', 'B', 10)
            self.set_text_color(79, 70, 229)
            self.set_fill_color(240, 240, 255)
            self.multi_cell(0, 6, text, fill=True)
            self.ln(2)

        def tabel(self, headers, data, col_widths=None):
            if col_widths is None:
                w = 190 / len(headers)
                col_widths = [w] * len(headers)

            # Header
            self.set_font('Helvetica', 'B', 9)
            self.set_fill_color(79, 70, 229)
            self.set_text_color(255, 255, 255)
            for i, h in enumerate(headers):
                self.cell(col_widths[i], 7, h, border=1, fill=True, align='C')
            self.ln()

            # Data
            self.set_font('Helvetica', '', 9)
            self.set_text_color(50, 50, 50)
            fill = False
            for row in data:
                if self.get_y() > 265:
                    self.add_page()
                    self.set_font('Helvetica', 'B', 9)
                    self.set_fill_color(79, 70, 229)
                    self.set_text_color(255, 255, 255)
                    for i, h in enumerate(headers):
                        self.cell(col_widths[i], 7, h, border=1, fill=True, align='C')
                    self.ln()
                    self.set_font('Helvetica', '', 9)
                    self.set_text_color(50, 50, 50)

                if fill:
                    self.set_fill_color(245, 245, 255)
                else:
                    self.set_fill_color(255, 255, 255)
                for i, cell in enumerate(row):
                    self.cell(col_widths[i], 6, str(cell), border=1, fill=True, align='C')
                self.ln()
                fill = not fill

        def highlight_box(self, text, color='blue'):
            if color == 'blue':
                self.set_fill_color(219, 234, 254)
                self.set_text_color(30, 64, 175)
            elif color == 'green':
                self.set_fill_color(220, 252, 231)
                self.set_text_color(22, 101, 52)
            elif color == 'yellow':
                self.set_fill_color(254, 249, 195)
                self.set_text_color(133, 77, 14)
            elif color == 'red':
                self.set_fill_color(254, 226, 226)
                self.set_text_color(153, 27, 27)

            self.set_font('Helvetica', '', 9)
            self.multi_cell(190, 5.5, text, fill=True)
            self.ln(2)

    # ================================================================
    # Build PDF
    # ================================================================
    pdf = LaporanPDF()
    pdf.alias_nb_pages()

    # ===== HALAMAN COVER =====
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font('Helvetica', 'B', 28)
    pdf.set_text_color(79, 70, 229)
    pdf.cell(0, 15, 'LAPORAN ANALISIS', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, 'Mini Search Engine', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_draw_color(79, 70, 229)
    pdf.set_line_width(1)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, 'Mata Kuliah: Temu Kembali Informasi', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, 'Ujian Tengah Semester (UTS)', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, 'Topik Dataset: Komentar Sosial Media tentang Imunisasi Balita', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f'Jumlah Dokumen: {N}', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f'Jumlah Term Unik: {len(engine.vocabulary)}', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 7, 'Teknologi: Python | Sastrawi Stemmer | TF-IDF | Cosine Similarity', align='C', new_x="LMARGIN", new_y="NEXT")

    # ===== DAFTAR ISI =====
    pdf.add_page()
    pdf.judul_bagian('DAFTAR ISI')
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(50, 50, 50)
    toc = [
        ('Pendahuluan', '3'),
        ('   Latar Belakang & Tujuan', '3'),
        ('   Metodologi Sistem', '3'),
        ('   Deskripsi Dataset', '3'),
        ('   Contoh Pre-processing', '4'),
        ('A. Analisis Bobot (IDF)', '5'),
        ('   A.1 Konsep Dasar IDF', '5'),
        ('   A.2 Perhitungan IDF: "imunisasi"', '5'),
        ('   A.3 Perhitungan IDF: "kejang"', '6'),
        ('   A.4 Perbandingan & Analisis Mendalam', '6'),
        ('B. Analisis Efek Normalisasi', '7'),
        ('   B.1 Mengapa Normalisasi Diperlukan?', '7'),
        ('   B.2 Hasil Dengan Cosine Normalization', '7'),
        ('   B.3 Hasil Tanpa Normalisasi', '8'),
        ('   B.4 Perbandingan & Analisis Mendalam', '8'),
        ('C. Evaluasi Sistem', '9'),
        ('   C.1 Konsep Evaluasi IR', '9'),
        ('   C.2 Skenario Query 1: "demam setelah imunisasi"', '9'),
        ('   C.3 Skenario Query 2: "vaksin anak balita"', '10'),
        ('   C.4 Analisis Kualitas Keseluruhan', '11'),
        ('Kesimpulan & Saran', '12'),
    ]
    for item, page in toc:
        dots = '.' * (70 - len(item))
        pdf.cell(0, 7, f'{item} {dots} {page}', new_x="LMARGIN", new_y="NEXT")

    # ===================================================================
    # PENDAHULUAN
    # ===================================================================
    pdf.add_page()
    pdf.judul_bagian('Pendahuluan')

    pdf.sub_judul('Latar Belakang dan Tujuan')
    pdf.paragraf(
        'Di era digital saat ini, volume informasi yang tersedia di internet berkembang sangat pesat. '
        'Bayangkan ketika seseorang ingin mencari informasi tentang "demam setelah imunisasi" dari '
        'ratusan komentar di media sosial. Membaca satu per satu tentu tidak efisien. Di sinilah '
        'peran sistem Temu Kembali Informasi (Information Retrieval) menjadi sangat penting.'
    )
    pdf.paragraf(
        'Laporan ini menyajikan hasil analisis dari Mini Search Engine yang kami bangun untuk '
        'mencari dokumen yang relevan dari koleksi komentar media sosial tentang imunisasi balita. '
        'Tujuan utama laporan ini adalah:'
    )
    pdf.paragraf(
        '1. Menganalisis bagaimana bobot IDF mempengaruhi proses pencarian, dengan membandingkan '
        'kata yang sering muncul ("imunisasi") dan yang jarang muncul ("kejang").\n'
        '2. Menunjukkan pentingnya normalisasi (Cosine Similarity) agar dokumen panjang tidak '
        'selalu menang dari dokumen pendek yang justru lebih relevan.\n'
        '3. Mengevaluasi kualitas sistem dengan menghitung Precision, Recall, dan F-Measure.'
    )

    pdf.sub_judul('Metodologi Sistem')
    pdf.paragraf(
        'Sistem Mini Search Engine ini bekerja melalui 4 tahap utama yang saling berkaitan. '
        'Berikut adalah penjelasan singkat setiap tahap beserta perannya:'
    )

    pdf.highlight_box(
        'ALUR KERJA SISTEM (PIPELINE):\n\n'
        'TAHAP 1 - PRE-PROCESSING (Pembersihan Teks)\n'
        'Teks mentah dari komentar sosial media tidak bisa langsung diproses karena mengandung '
        'huruf besar-kecil yang bercampur, tanda baca, emoji, dan bahasa slang. Oleh karena itu, '
        'teks harus dibersihkan terlebih dahulu melalui 5 langkah:\n'
        '  a) Case Folding: Mengubah semua huruf menjadi huruf kecil agar "Imunisasi" dan '
        '"imunisasi" dianggap kata yang sama.\n'
        '  b) Punctuation Removal: Menghapus tanda baca, angka, emoji, dan URL.\n'
        '  c) Tokenisasi: Memecah kalimat menjadi kata-kata individual (token).\n'
        '  d) Stopword Removal: Menghapus kata-kata umum yang tidak bermakna seperti "yang", '
        '"dan", "di", "ke", karena kata-kata ini muncul di hampir semua dokumen dan tidak '
        'membantu membedakan satu dokumen dengan yang lain.\n'
        '  e) Stemming: Mengubah kata ke bentuk dasarnya menggunakan library Sastrawi, '
        'misalnya "membahas" menjadi "bahas", "imunisasi" tetap "imunisasi".\n\n'
        'TAHAP 2 - INDEXING (Pembuatan Inverted Index)\n'
        'Setelah pre-processing, sistem membangun Inverted Index, yaitu struktur data yang '
        'memetakan setiap kata (term) ke daftar dokumen yang mengandungnya. Ini seperti '
        'indeks di belakang buku: kita bisa langsung mencari kata dan mengetahui halaman '
        'mana saja yang memuat kata tersebut.\n\n'
        'TAHAP 3 - TF-IDF WEIGHTING (Pembobotan)\n'
        'Setiap term diberi bobot berdasarkan dua faktor:\n'
        '  - TF (Term Frequency): Seberapa sering kata muncul di suatu dokumen. Semakin sering '
        'muncul, semakin penting kata itu untuk dokumen tersebut.\n'
        '  - IDF (Inverse Document Frequency): Seberapa jarang kata muncul di seluruh koleksi. '
        'Kata yang jarang muncul dianggap lebih informatif dan bernilai tinggi.\n\n'
        'TAHAP 4 - RETRIEVAL (Pencarian dengan Cosine Similarity)\n'
        'Ketika pengguna memasukkan query, sistem juga memproses query dengan cara yang sama, '
        'lalu menghitung kemiripan antara vektor query dan vektor setiap dokumen menggunakan '
        'Cosine Similarity. Dokumen diurutkan dari yang paling mirip.',
        'blue'
    )

    pdf.sub_judul('Deskripsi Dataset')
    pdf.paragraf(
        f'Dataset yang digunakan dalam penelitian ini berisi {N} komentar dari media sosial '
        '(terutama TikTok dan YouTube) yang membahas topik imunisasi pada balita. Komentar-komentar '
        'ini dipilih karena mencerminkan bahasa sehari-hari masyarakat Indonesia yang sering '
        'menggunakan singkatan, bahasa gaul, dan ejaan tidak baku.'
    )
    pdf.paragraf(
        'Beberapa contoh tantangan dalam dataset ini:\n'
        '- Penggunaan singkatan: "gk" = "tidak", "krna" = "karena", "yg" = "yang"\n'
        '- Bahasa informal: "aku mah", "bunda", "bun", "kak"\n'
        '- Campuran bahasa Indonesia dan istilah medis: "DPT", "BCG", "paracetamol"\n'
        '- Ejaan tidak baku: "imunasi" (seharusnya "imunisasi"), "msk" (seharusnya "masuk")\n\n'
        f'Setelah proses pre-processing, dari {N} dokumen ini ditemukan {len(engine.vocabulary)} '
        'term unik (vocabulary) yang digunakan untuk membangun indeks pencarian.'
    )

    pdf.tabel(
        ['Properti Dataset', 'Nilai'],
        [
            ['Total Dokumen', str(N)],
            ['Sumber Data', 'TikTok, YouTube (komentar)'],
            ['Topik', 'Imunisasi Balita'],
            ['Bahasa', 'Indonesia (informal/slang)'],
            ['Jumlah Term Unik', str(len(engine.vocabulary))],
            ['Library Stemming', 'Sastrawi (Bahasa Indonesia)'],
        ],
        [70, 120]
    )

    # --- Contoh Pre-processing ---
    pdf.add_page()
    pdf.sub_judul('Contoh Pre-processing Langkah demi Langkah')
    pdf.paragraf(
        f'Untuk memahami bagaimana sistem memproses teks, berikut ditampilkan contoh '
        f'pre-processing pada Dokumen D{example_doc_id}. Proses ini dilakukan pada SEMUA '
        'dokumen dalam koleksi sebelum pencarian bisa dilakukan.'
    )

    # Tampilkan contoh preprocessing step by step
    pdf.paragraf(f'Teks asli Dokumen D{example_doc_id}:')
    pdf.highlight_box(clean_for_pdf(example_text[:300]) + ('...' if len(example_text) > 300 else ''), 'blue')

    pdf.paragraf('Langkah 1 - Case Folding (semua huruf menjadi kecil):')
    cf_text = clean_for_pdf(example_steps['case_folding'][:200]) + '...'
    pdf.highlight_box(cf_text, 'yellow')

    pdf.paragraf('Langkah 2 - Punctuation Removal (hapus tanda baca, angka, emoji):')
    pr_text = clean_for_pdf(example_steps['punctuation_removal'][:200]) + '...'
    pdf.highlight_box(pr_text, 'yellow')

    pdf.paragraf(f'Langkah 3 - Tokenisasi (pecah menjadi {len(example_steps["tokenization"])} token):')
    tok_preview = ', '.join(clean_for_pdf(t) for t in example_steps['tokenization'][:15])
    pdf.highlight_box(tok_preview + ', ...', 'yellow')

    pdf.paragraf(f'Langkah 4 - Stopword Removal (tersisa {len(example_steps["stopword_removal"])} token):')
    sw_preview = ', '.join(clean_for_pdf(t) for t in example_steps['stopword_removal'][:15])
    pdf.highlight_box(sw_preview + ('...' if len(example_steps['stopword_removal']) > 15 else ''), 'yellow')

    pdf.paragraf(f'Langkah 5 - Stemming (hasil akhir {len(example_steps["stemming"])} token):')
    st_preview = ', '.join(clean_for_pdf(t) for t in example_steps['stemming'][:15])
    pdf.highlight_box(st_preview + ('...' if len(example_steps['stemming']) > 15 else ''), 'yellow')

    orig_count = len(example_steps['tokenization'])
    final_count = len(example_steps['stemming'])
    reduction = orig_count - final_count
    pct = (reduction / orig_count * 100) if orig_count > 0 else 0

    pdf.highlight_box(
        f'RINGKASAN PRE-PROCESSING DOKUMEN D{example_doc_id}:\n'
        f'- Token awal (sebelum filtering): {orig_count}\n'
        f'- Token akhir (setelah stopword + stemming): {final_count}\n'
        f'- Pengurangan: {reduction} token ({pct:.0f}%)\n\n'
        'Artinya, lebih dari separuh kata dalam teks asli adalah kata-kata umum yang tidak '
        'bermakna untuk pencarian. Pre-processing berhasil menyaring hanya kata-kata penting '
        'yang benar-benar merepresentasikan isi dokumen.',
        'green'
    )

    # ===================================================================
    # BAGIAN A: ANALISIS BOBOT IDF
    # ===================================================================
    pdf.add_page()
    pdf.judul_bagian('A. Analisis Bobot (IDF)')

    pdf.sub_judul('A.1 Konsep Dasar IDF')
    pdf.paragraf(
        'Sebelum masuk ke perhitungan, mari kita pahami dulu mengapa IDF itu penting. '
        'Bayangkan kita sedang mencari artikel tentang "kejang setelah imunisasi" di sebuah '
        'kumpulan komentar tentang imunisasi. Hampir semua komentar akan mengandung kata '
        '"imunisasi" karena memang itu topik utamanya. Tapi hanya sedikit komentar yang '
        'membahas "kejang". Jadi, kata mana yang lebih berguna untuk menemukan komentar '
        'yang spesifik? Tentu saja "kejang", karena kata ini bisa menyaring dokumen secara '
        'lebih tajam.'
    )
    pdf.paragraf(
        'IDF (Inverse Document Frequency) menangkap intuisi ini secara matematis. '
        'Rumusnya sangat sederhana namun sangat efektif:'
    )
    pdf.rumus(
        '  IDF(t) = log10(N / df(t))\n\n'
        '  dimana:\n'
        '  N    = jumlah total dokumen dalam koleksi\n'
        '  df(t)= jumlah dokumen yang mengandung term t\n\n'
        '  Semakin BANYAK dokumen yang mengandung term t,\n'
        '  maka df(t) besar, N/df(t) kecil, log-nya kecil.\n'
        '  => Kata UMUM mendapat bobot RENDAH.\n\n'
        '  Semakin SEDIKIT dokumen yang mengandung term t,\n'
        '  maka df(t) kecil, N/df(t) besar, log-nya besar.\n'
        '  => Kata JARANG mendapat bobot TINGGI.'
    )
    pdf.paragraf(
        'Analoginya seperti ini: Jika semua mahasiswa di kelas memakai baju putih, '
        'maka "baju putih" tidak bisa membedakan siapa yang kita cari. Tapi jika hanya '
        'satu orang yang memakai topi merah, maka "topi merah" langsung mempersempit pencarian. '
        'Itulah esensi IDF: kata yang jarang muncul memiliki daya pembeda yang lebih kuat.'
    )

    # --- A.2 Keyword 1: imunisasi ---
    pdf.sub_judul(f'A.2 Perhitungan IDF Kata Kunci: "{keyword1}"')

    pdf.paragraf(
        f'Kata "{keyword1}" dipilih karena merupakan topik utama dataset. Hampir semua '
        'komentar dalam koleksi ini membahas tentang imunisasi, sehingga kata ini diharapkan '
        'muncul di banyak dokumen dan memiliki IDF yang rendah.'
    )

    pdf.paragraf(f'Data yang diketahui:\n'
                 f'- N (total dokumen dalam koleksi) = {N}\n'
                 f'- df("{keyword1}") = {df1} dokumen yang mengandung kata ini\n'
                 f'- Persentase kemunculan: {df1}/{N} = {df1/N*100:.1f}% dari seluruh koleksi')

    pdf.paragraf(f'Dokumen yang mengandung "{keyword1}": {", ".join(["D"+str(d) for d in docs_kw1])}')

    pdf.paragraf('Perhitungan langkah demi langkah:')
    pdf.rumus(f'  IDF("{keyword1}") = log10(N / df("{keyword1}"))\n'
              f'                   = log10({N} / {df1})\n'
              f'                   = log10({N/df1:.6f})\n'
              f'                   = {idf1:.6f}')

    pdf.highlight_box(
        f'HASIL: IDF("{keyword1}") = {idf1:.4f}\n\n'
        f'Interpretasi: Kata "{keyword1}" muncul di {df1} dari {N} dokumen ({df1/N*100:.1f}%). '
        f'Ini berarti kata ini sangat UMUM dalam koleksi kita. Nilai IDF-nya rendah ({idf1:.4f}), '
        'yang menunjukkan bahwa kata ini KURANG BERGUNA untuk membedakan satu dokumen dari '
        'dokumen lainnya. Hal ini masuk akal karena dataset kita memang bertopik imunisasi, '
        'jadi hampir semua komentar akan menyebut kata "imunisasi".',
        'blue'
    )

    # --- A.3 Keyword 2: kejang ---
    pdf.sub_judul(f'A.3 Perhitungan IDF Kata Kunci: "{keyword2}"')

    pdf.paragraf(
        f'Kata "{keyword2}" dipilih sebagai kontras dari "{keyword1}". Tidak semua komentar '
        f'membahas tentang kejang, kata ini hanya muncul di komentar-komentar spesifik yang '
        'menceritakan pengalaman anak mengalami kejang setelah imunisasi. Kita menduga kata ini '
        'akan memiliki IDF yang lebih tinggi.'
    )

    pdf.paragraf(f'Data yang diketahui:\n'
                 f'- N (total dokumen dalam koleksi) = {N}\n'
                 f'- df("{keyword2}") = {df2} dokumen yang mengandung kata ini\n'
                 f'- Persentase kemunculan: {df2}/{N} = {df2/N*100:.1f}% dari seluruh koleksi')

    pdf.paragraf(f'Dokumen yang mengandung "{keyword2}": {", ".join(["D"+str(d) for d in docs_kw2])}')

    pdf.paragraf('Perhitungan langkah demi langkah:')
    pdf.rumus(f'  IDF("{keyword2}") = log10(N / df("{keyword2}"))\n'
              f'                  = log10({N} / {df2})\n'
              f'                  = log10({N/df2:.6f})\n'
              f'                  = {idf2:.6f}')

    pdf.highlight_box(
        f'HASIL: IDF("{keyword2}") = {idf2:.4f}\n\n'
        f'Interpretasi: Kata "{keyword2}" hanya muncul di {df2} dari {N} dokumen ({df2/N*100:.1f}%). '
        f'Nilai IDF-nya jauh lebih tinggi ({idf2:.4f}) dibandingkan "{keyword1}" ({idf1:.4f}). '
        'Ini menunjukkan bahwa kata "kejang" LEBIH INFORMATIF dan LEBIH BERGUNA untuk pencarian, '
        'karena bisa langsung mengarahkan kita ke dokumen-dokumen spesifik yang membahas '
        'tentang kejang pasca imunisasi.',
        'blue'
    )

    # --- A.4 Perbandingan ---
    pdf.add_page()
    pdf.sub_judul('A.4 Perbandingan dan Analisis Mendalam')

    pdf.tabel(
        ['Aspek', f'"{keyword1}"', f'"{keyword2}"'],
        [
            ['Document Frequency (df)', str(df1), str(df2)],
            ['% Dokumen', f'{df1/N*100:.1f}%', f'{df2/N*100:.1f}%'],
            ['Nilai IDF', f'{idf1:.4f}', f'{idf2:.4f}'],
            ['Kategori', 'Kata Umum', 'Kata Spesifik'],
            ['Daya Pembeda', 'Rendah', 'Tinggi'],
        ],
        [50, 70, 70]
    )
    pdf.ln(3)

    higher_kw = keyword2 if idf2 > idf1 else keyword1
    lower_kw = keyword1 if idf2 > idf1 else keyword2
    higher_idf = max(idf1, idf2)
    lower_idf = min(idf1, idf2)

    pdf.paragraf(
        f'Dari tabel di atas, jelas terlihat perbedaan yang sangat signifikan antara kedua kata. '
        f'Kata "{higher_kw}" memiliki IDF {higher_idf:.4f}, sedangkan "{lower_kw}" hanya {lower_idf:.4f}. '
        f'Selisihnya adalah {abs(idf2-idf1):.4f}, yang berarti kata "{higher_kw}" bernilai '
        f'sekitar {higher_idf/lower_idf:.1f}x lebih informatif dibandingkan "{lower_kw}".'
    )

    pdf.highlight_box(
        'MENGAPA KATA JARANG LEBIH BERHARGA? - PENJELASAN LENGKAP\n\n'
        'Untuk memahami konsep ini, mari gunakan analogi kehidupan sehari-hari:\n\n'
        '1. ANALOGI PERPUSTAKAAN:\n'
        'Bayangkan Anda di perpustakaan kampus dan mencari buku tentang "efek samping '
        'vaksin yang menyebabkan kejang pada bayi". Jika Anda hanya mencari dengan kata '
        '"buku" maka semua item di perpustakaan akan cocok, karena semuanya memang buku. '
        'Kata "buku" tidak membantu sama sekali. Tapi jika Anda mencari "kejang", hanya '
        'sedikit buku yang akan muncul, dan kemungkinan besar itu buku yang Anda butuhkan.\n\n'
        '2. PRINSIP TEORI INFORMASI:\n'
        'Claude Shannon, bapak teori informasi, menunjukkan bahwa peristiwa yang jarang '
        'terjadi membawa lebih banyak informasi daripada peristiwa yang sudah diharapkan. '
        'Jika kita sudah tahu semua komentar membahas "imunisasi", maka kemunculan kata '
        '"imunisasi" tidak memberikan informasi baru. Tapi kemunculan kata "kejang" langsung '
        'memberikan sinyal bahwa komentar tersebut membahas topik yang spesifik.\n\n'
        '3. DAMPAK PADA PENCARIAN:\n'
        'Dalam konteks mesin pencari, kata dengan IDF tinggi membantu MEMPERSEMPIT hasil '
        'pencarian ke dokumen yang paling relevan. Ketika pengguna mencari "kejang setelah '
        'imunisasi", kata "kejang" (IDF tinggi) akan menjadi pembeda utama, sedangkan kata '
        '"imunisasi" (IDF rendah) hanya berfungsi sebagai filter tambahan.\n\n'
        '4. MENGAPA PAKAI LOGARITMA?\n'
        'Rumus IDF menggunakan log10 untuk "meredam" perbedaan yang terlalu ekstrem. '
        'Tanpa logaritma, kata yang muncul di 1 dokumen vs 50 dokumen akan memiliki '
        'rasio 50:1, yang terlalu dominan. Logaritma membuat perbedaan ini lebih proporsional '
        'dan seimbang saat dikombinasikan dengan TF.',
        'green'
    )

    # ===================================================================
    # BAGIAN B: ANALISIS EFEK NORMALISASI
    # ===================================================================
    pdf.add_page()
    pdf.judul_bagian('B. Analisis Efek Normalisasi')

    pdf.sub_judul('B.1 Mengapa Normalisasi Diperlukan?')
    pdf.paragraf(
        'Sebelum membandingkan hasil, penting untuk memahami masalah yang diselesaikan '
        'oleh normalisasi. Dalam Vector Space Model, setiap dokumen dan query '
        'direpresentasikan sebagai vektor di ruang berdimensi banyak. Setiap dimensi '
        'mewakili satu term, dan nilainya adalah bobot TF-IDF term tersebut.'
    )

    pdf.paragraf(
        'MASALAH tanpa normalisasi: Jika kita hanya menggunakan Dot Product (perkalian titik) '
        'sebagai ukuran kemiripan, maka dokumen yang panjang (mengandung banyak kata) secara '
        'alami akan mendapat skor lebih tinggi, bukan karena lebih relevan, tetapi semata-mata '
        'karena vektornya lebih "besar" (magnitude lebih besar).'
    )

    pdf.paragraf(
        'ANALOGI: Bayangkan ada dua toko. Toko A kecil tapi khusus menjual obat demam anak. '
        'Toko B adalah mall besar yang menjual segalanya, termasuk obat demam di pojok kecil. '
        'Jika kita mencari "obat demam anak", Toko A jelas lebih relevan. Tapi tanpa normalisasi, '
        'Toko B (yang besar) bisa mendapat skor lebih tinggi hanya karena ukurannya. '
        'Cosine Similarity mengatasi masalah ini.'
    )

    pdf.rumus(
        '  TANPA NORMALISASI (Dot Product):\n'
        '  Score(q,d) = q . d = sum(q_i x d_i)\n'
        '  => Bias terhadap dokumen panjang!\n\n'
        '  DENGAN NORMALISASI (Cosine Similarity):\n'
        '  Score(q,d) = (q . d) / (|q| x |d|)\n'
        '  => Mengukur SUDUT antara vektor, bukan besarnya.\n'
        '  => Skor selalu antara 0 (tidak mirip) dan 1 (identik).'
    )

    # --- B.2 Dengan Normalisasi ---
    pdf.sub_judul('B.2 Hasil Dengan Cosine Normalization')
    pdf.paragraf(f'Query yang digunakan: "{query_norm}"\n'
                 f'Token setelah pre-processing: {", ".join(tokens_norm)}')
    pdf.paragraf(
        'Pada tabel berikut, perhatikan bahwa skor berada di rentang 0-1 (hasil normalisasi). '
        'Kolom "Token" menunjukkan jumlah kata bermakna dalam dokumen setelah pre-processing.'
    )

    data_norm = []
    for i, r in enumerate(results_norm[:10]):
        data_norm.append([
            str(i+1),
            f'D{r["doc_id"]}',
            f'{r["score"]:.6f}',
            str(r['token_count']),
            clean_for_pdf(r['text'][:45]) + '...'
        ])

    pdf.tabel(
        ['Rank', 'Doc', 'Score', 'Token', 'Komentar (preview)'],
        data_norm,
        [12, 14, 28, 14, 122]
    )

    # --- B.3 Tanpa Normalisasi ---
    pdf.ln(3)
    pdf.sub_judul('B.3 Hasil Tanpa Normalisasi (Dot Product Saja)')
    pdf.paragraf(
        'Pada tabel berikut, skor TIDAK dinormalisasi (hanya dot product). Perhatikan bahwa '
        'skor bisa sangat besar dan tidak terbatas pada rentang 0-1. Dokumen dengan token '
        'lebih banyak cenderung mendapat skor lebih tinggi.'
    )

    data_no_norm = []
    for i, r in enumerate(results_no_norm[:10]):
        data_no_norm.append([
            str(i+1),
            f'D{r["doc_id"]}',
            f'{r["score"]:.6f}',
            str(r['token_count']),
            clean_for_pdf(r['text'][:45]) + '...'
        ])

    pdf.tabel(
        ['Rank', 'Doc', 'Score', 'Token', 'Komentar (preview)'],
        data_no_norm,
        [12, 14, 28, 14, 122]
    )

    # --- B.4 Perbandingan ---
    pdf.ln(3)
    pdf.sub_judul('B.4 Perbandingan dan Analisis Mendalam')

    pdf.paragraf(
        'Tabel berikut membandingkan peringkat dokumen antara kedua metode. Kolom "Perubahan" '
        'menunjukkan apakah peringkat dokumen naik atau turun ketika normalisasi digunakan. '
        'Perhatikan bagaimana dokumen dengan jumlah token yang berbeda mengalami perubahan peringkat.'
    )

    rank_norm = {r['doc_id']: i+1 for i, r in enumerate(results_norm[:10])}
    rank_no_norm = {r['doc_id']: i+1 for i, r in enumerate(results_no_norm[:10])}
    token_counts = {r['doc_id']: r['token_count'] for r in results_norm[:10]}
    token_counts.update({r['doc_id']: r['token_count'] for r in results_no_norm[:10]})

    all_docs_compared = set(list(rank_norm.keys())[:5] + list(rank_no_norm.keys())[:5])
    compare_data = []
    for doc_id in sorted(all_docs_compared):
        rn = rank_norm.get(doc_id, '-')
        rnn = rank_no_norm.get(doc_id, '-')
        tc = token_counts.get(doc_id, '-')
        change = ''
        if isinstance(rn, int) and isinstance(rnn, int):
            diff = rnn - rn
            if diff > 0:
                change = f'Naik {diff}'
            elif diff < 0:
                change = f'Turun {abs(diff)}'
            else:
                change = 'Tetap'
        compare_data.append([f'D{doc_id}', str(tc), str(rn), str(rnn), change])

    pdf.tabel(
        ['Dokumen', 'Jml Token', 'Rank (Norm)', 'Rank (No Norm)', 'Perubahan'],
        compare_data,
        [25, 25, 35, 35, 70]
    )

    pdf.ln(3)

    pdf.highlight_box(
        'ANALISIS MENDALAM - MENGAPA NORMALISASI MENGUBAH PERINGKAT:\n\n'
        '1. MASALAH BIAS PANJANG DOKUMEN:\n'
        'Tanpa normalisasi, skor dihitung sebagai dot product antara vektor query dan '
        'vektor dokumen. Ini sama seperti menjumlahkan perkalian bobot term yang cocok. '
        'Dokumen yang lebih panjang memiliki lebih banyak term, sehingga lebih banyak '
        'peluang term query yang cocok. Akibatnya, dokumen panjang selalu diuntungkan '
        'meskipun isinya tidak terlalu fokus pada topik query.\n\n'
        'Contoh nyata: Sebuah komentar panjang yang membahas banyak hal (imunisasi, harga, '
        'jadwal, dll) bisa mendapat skor tinggi hanya karena mengandung banyak kata, '
        'padahal komentar pendek seperti "demam setelah imunisasi di namakan KIPI" mungkin '
        'jauh lebih relevan untuk query "demam setelah imunisasi".\n\n'
        '2. SOLUSI COSINE SIMILARITY:\n'
        'Cosine Similarity membagi dot product dengan perkalian panjang (magnitude) kedua '
        'vektor. Ini menghilangkan pengaruh "ukuran" dan hanya mengukur "arah" vektor. '
        'Dua vektor yang mengarah ke arah yang sama mendapat skor tinggi, terlepas dari '
        'seberapa panjang vektornya.\n\n'
        'Secara visual, bayangkan dua anak panah. Cosine Similarity mengukur seberapa '
        'kecil sudut antara keduanya. Anak panah pendek yang searah dengan query tetap '
        'mendapat skor tinggi, sedangkan anak panah panjang yang agak melenceng mendapat '
        'skor lebih rendah.\n\n'
        '3. KESIMPULAN PRAKTIS:\n'
        'Cosine Normalization membuat sistem pencarian menjadi lebih ADIL karena:\n'
        '- Dokumen pendek yang sangat relevan tidak kalah dari dokumen panjang yang kurang fokus\n'
        '- Skor menjadi sebanding (0-1) sehingga mudah diinterpretasikan\n'
        '- Kualitas pencarian meningkat karena peringkat lebih mencerminkan relevansi sesungguhnya',
        'yellow'
    )

    # ===================================================================
    # BAGIAN C: EVALUASI SISTEM
    # ===================================================================
    pdf.add_page()
    pdf.judul_bagian('C. Evaluasi Sistem')

    pdf.sub_judul('C.1 Konsep Evaluasi dalam Information Retrieval')
    pdf.paragraf(
        'Untuk mengetahui apakah mesin pencari yang kita bangun benar-benar bagus, kita perlu '
        'mengevaluasinya secara objektif. Caranya adalah dengan membandingkan hasil pencarian '
        'sistem (apa yang ditemukan mesin) dengan penilaian manusia (apa yang sebenarnya relevan). '
        'Dalam dunia Information Retrieval, ada tiga metrik utama yang digunakan:'
    )

    pdf.highlight_box(
        'TIGA METRIK EVALUASI UTAMA:\n\n'
        '1. PRECISION (Ketepatan) = "Dari yang DITEMUKAN sistem, berapa yang BENAR relevan?"\n'
        '   Rumus: Precision = |Relevan dan Ditemukan| / |Total Ditemukan|\n'
        '   Contoh: Jika sistem menemukan 10 dokumen dan 7 di antaranya relevan,\n'
        '   maka Precision = 7/10 = 0.70 (70%).\n'
        '   Precision tinggi = sedikit "sampah" di hasil pencarian.\n\n'
        '2. RECALL (Kelengkapan) = "Dari yang BENAR relevan, berapa yang BERHASIL ditemukan?"\n'
        '   Rumus: Recall = |Relevan dan Ditemukan| / |Total Relevan|\n'
        '   Contoh: Jika ada 12 dokumen relevan total dan sistem menemukan 5 di antaranya,\n'
        '   maka Recall = 5/12 = 0.42 (42%).\n'
        '   Recall tinggi = sedikit dokumen relevan yang terlewat.\n\n'
        '3. F-MEASURE (Keseimbangan) = Rata-rata harmonis Precision dan Recall.\n'
        '   Rumus: F = 2 x (P x R) / (P + R)\n'
        '   Mengapa harmonic mean? Karena jika salah satu dari P atau R sangat rendah,\n'
        '   F-Measure juga akan rendah. Ini memastikan sistem harus baik di KEDUA aspek.\n\n'
        'GROUND TRUTH: Untuk menghitung metrik ini, kita perlu menentukan terlebih dahulu\n'
        'dokumen mana saja yang BENAR-BENAR relevan menurut penilaian manusia. Ini disebut\n'
        'Ground Truth, yaitu "jawaban yang benar" yang menjadi acuan evaluasi.',
        'blue'
    )

    # --- C.2 Query 1 ---
    pdf.sub_judul(f'C.2 Skenario Query 1: "{query1}"')
    pdf.paragraf(f'Token setelah pre-processing: {", ".join(tokens_q1)}')

    pdf.paragraf(
        'Penentuan Ground Truth:\n'
        'Untuk query "demam setelah imunisasi", kami membaca satu per satu seluruh 50 dokumen '
        'dalam koleksi dan menilai secara manual apakah setiap dokumen benar-benar membahas '
        'tentang demam sebagai efek/reaksi dari imunisasi. Dokumen dianggap relevan jika '
        'memenuhi salah satu kriteria berikut:\n'
        '- Menceritakan pengalaman anak demam setelah disuntik imunisasi\n'
        '- Memberikan tips menangani demam pasca imunisasi\n'
        '- Bertanya atau berdiskusi tentang demam setelah imunisasi\n'
        '- Menyebutkan demam sebagai efek samping (KIPI) dari imunisasi'
    )

    pdf.paragraf(
        f'Hasil penilaian: ditemukan {len(ground_truth_q1)} dokumen yang relevan:\n'
        f'{", ".join(["D"+str(d) for d in sorted(ground_truth_q1)])}'
    )

    # Detail ground truth Q1
    gt_q1_reasons = {
        8: 'Menceritakan kompres hangat & parasetamol untuk demam pasca imunisasi',
        10: 'Menceritakan anak suntik DPT tidak sampai demam (membahas topik demam)',
        13: 'Tidak lanjut imunisasi karena takut demam & rewel',
        15: 'Belum imunisasi 18 bulan karena kemarin demam',
        22: 'Bertanya cara menangani bayi demam setelah imunisasi',
        27: 'Anak kejang dan demam setelah suntik',
        31: 'Menyuruh taat imunisasi jangan takut demam',
        34: 'Tips imunisasi: siapkan obat penurun demam',
        40: 'Anak sakit 2 hari setelah imunisasi (demam)',
        42: 'Lengkap membahas cara menangani demam pasca vaksin',
        45: 'Membahas panas/kejang setelah imunisasi',
        46: 'Membahas demam setelah imunisasi (KIPI)',
    }

    gt_data_q1 = []
    for did in sorted(ground_truth_q1):
        reason = gt_q1_reasons.get(did, '-')
        gt_data_q1.append([f'D{did}', clean_for_pdf(reason)])

    pdf.tabel(
        ['Dokumen', 'Alasan Dianggap Relevan'],
        gt_data_q1,
        [20, 170]
    )

    pdf.ln(3)

    pdf.paragraf('Hasil pencarian sistem (Top-10):')

    q1_data = []
    for i, r in enumerate(results_q1[:10]):
        rel = 'Ya' if r['doc_id'] in ground_truth_q1 else 'Tidak'
        q1_data.append([
            str(i+1),
            f'D{r["doc_id"]}',
            f'{r["score"]:.6f}',
            rel,
            clean_for_pdf(r['text'][:50]) + '...'
        ])

    pdf.tabel(
        ['Rank', 'Doc', 'Score', 'Relevan?', 'Preview'],
        q1_data,
        [12, 14, 28, 18, 118]
    )

    pdf.ln(3)

    pdf.paragraf('Perhitungan metrik evaluasi langkah demi langkah:')

    # False positives dan false negatives
    fp1 = ret1 - rr1
    fn1 = ground_truth_q1 - rr1

    pdf.rumus(
        f'  Langkah 1: Identifikasi himpunan\n'
        f'  Retrieved (Top-10)       = {{{", ".join(["D"+str(d) for d in sorted(ret1)])}}}\n'
        f'  Relevant (Ground Truth)  = {{{", ".join(["D"+str(d) for d in sorted(ground_truth_q1)])}}}\n\n'
        f'  Langkah 2: Cari irisan (dokumen yang relevan DAN ditemukan)\n'
        f'  Relevant n Retrieved     = {{{", ".join(["D"+str(d) for d in sorted(rr1)])}}}\n'
        f'  |Relevant n Retrieved|   = {len(rr1)}\n\n'
        f'  Langkah 3: Hitung metrik\n'
        f'  Precision = {len(rr1)} / {len(ret1)} = {p1:.4f} ({p1*100:.1f}%)\n'
        f'  Recall    = {len(rr1)} / {len(ground_truth_q1)} = {r1:.4f} ({r1*100:.1f}%)\n'
        f'  F-Measure = 2 x ({p1:.4f} x {r1:.4f}) / ({p1:.4f} + {r1:.4f}) = {f1:.4f} ({f1*100:.1f}%)'
    )

    pdf.highlight_box(
        f'INTERPRETASI QUERY 1:\n\n'
        f'Precision {p1*100:.1f}% berarti dari 10 dokumen yang ditemukan sistem, '
        f'{len(rr1)} di antaranya benar-benar relevan dan {len(fp1)} tidak relevan '
        f'(false positive).'
        + (f'\nDokumen false positive: {", ".join(["D"+str(d) for d in sorted(fp1)])} - '
           'dokumen ini ditemukan sistem tapi sebenarnya tidak relevan dengan query.' if fp1 else '')
        + f'\n\nRecall {r1*100:.1f}% berarti dari {len(ground_truth_q1)} dokumen yang '
        f'sebenarnya relevan, sistem berhasil menemukan {len(rr1)} dan melewatkan '
        f'{len(fn1)} dokumen (false negative).'
        + (f'\nDokumen yang terlewat: {", ".join(["D"+str(d) for d in sorted(fn1)])} - '
           'dokumen ini relevan tapi tidak masuk Top-10.' if fn1 else ''),
        'green'
    )

    # --- C.3 Query 2 ---
    pdf.add_page()
    pdf.sub_judul(f'C.3 Skenario Query 2: "{query2}"')
    pdf.paragraf(f'Token setelah pre-processing: {", ".join(tokens_q2)}')

    pdf.paragraf(
        'Penentuan Ground Truth:\n'
        'Untuk query "vaksin anak balita", kami kembali membaca seluruh 50 dokumen dan menilai '
        'secara manual. Dokumen dianggap relevan jika memenuhi salah satu kriteria:\n'
        '- Membahas prosedur atau jenis vaksin untuk anak/balita\n'
        '- Menjelaskan apa itu vaksin/imunisasi kepada anak\n'
        '- Bertanya tentang vaksin di klinik/posyandu untuk anak\n'
        '- Menceritakan pengalaman vaksinasi anak'
    )

    pdf.paragraf(
        f'Hasil penilaian: ditemukan {len(ground_truth_q2)} dokumen yang relevan:\n'
        f'{", ".join(["D"+str(d) for d in sorted(ground_truth_q2)])}'
    )

    # Detail ground truth Q2
    gt_q2_reasons = {
        1: 'Membahas pentingnya imunisasi anak usia 5 tahun',
        2: 'Bertanya prosedur vaksin anak dan uji lab',
        4: 'Membahas kompres setelah imunisasi (pengalaman vaksin anak)',
        5: 'Bertanya vaksin di klinik mana',
        7: 'Informasi jenis dan biaya vaksin imunisasi anak',
        28: 'Bertanya jumlah suntikan DPT untuk bayi',
        29: 'Menjelaskan fungsi imunisasi/vaksin untuk tubuh',
        32: 'Menjelaskan apa itu imunisasi/vaksin secara edukatif',
        38: 'Informasi tempat vaksinasi bayi (bidan/posyandu)',
        43: 'Menjelaskan proses imunisasi menyuntikkan bakteri lemah',
        44: 'Membahas perbedaan jenis vaksin mahal vs murah untuk anak',
        47: 'Membahas perbedaan DPT BPJS vs berbayar untuk anak',
    }

    gt_data_q2 = []
    for did in sorted(ground_truth_q2):
        reason = gt_q2_reasons.get(did, '-')
        gt_data_q2.append([f'D{did}', clean_for_pdf(reason)])

    pdf.tabel(
        ['Dokumen', 'Alasan Dianggap Relevan'],
        gt_data_q2,
        [20, 170]
    )

    pdf.ln(3)

    pdf.paragraf('Hasil pencarian sistem (Top-10):')

    q2_data = []
    for i, r in enumerate(results_q2[:10]):
        rel = 'Ya' if r['doc_id'] in ground_truth_q2 else 'Tidak'
        q2_data.append([
            str(i+1),
            f'D{r["doc_id"]}',
            f'{r["score"]:.6f}',
            rel,
            clean_for_pdf(r['text'][:50]) + '...'
        ])

    pdf.tabel(
        ['Rank', 'Doc', 'Score', 'Relevan?', 'Preview'],
        q2_data,
        [12, 14, 28, 18, 118]
    )

    pdf.ln(3)

    # False positives dan false negatives
    fp2 = ret2 - rr2
    fn2 = ground_truth_q2 - rr2

    pdf.paragraf('Perhitungan metrik evaluasi langkah demi langkah:')
    pdf.rumus(
        f'  Langkah 1: Identifikasi himpunan\n'
        f'  Retrieved (Top-10)       = {{{", ".join(["D"+str(d) for d in sorted(ret2)])}}}\n'
        f'  Relevant (Ground Truth)  = {{{", ".join(["D"+str(d) for d in sorted(ground_truth_q2)])}}}\n\n'
        f'  Langkah 2: Cari irisan\n'
        f'  Relevant n Retrieved     = {{{", ".join(["D"+str(d) for d in sorted(rr2)])}}}\n'
        f'  |Relevant n Retrieved|   = {len(rr2)}\n\n'
        f'  Langkah 3: Hitung metrik\n'
        f'  Precision = {len(rr2)} / {len(ret2)} = {p2:.4f} ({p2*100:.1f}%)\n'
        f'  Recall    = {len(rr2)} / {len(ground_truth_q2)} = {r2:.4f} ({r2*100:.1f}%)\n'
        f'  F-Measure = 2 x ({p2:.4f} x {r2:.4f}) / ({p2:.4f} + {r2:.4f}) = {f2:.4f} ({f2*100:.1f}%)'
    )

    pdf.highlight_box(
        f'INTERPRETASI QUERY 2:\n\n'
        f'Precision {p2*100:.1f}% berarti dari 10 dokumen yang ditemukan sistem, '
        f'{len(rr2)} di antaranya benar-benar relevan dan {len(fp2)} tidak relevan.'
        + (f'\nDokumen false positive: {", ".join(["D"+str(d) for d in sorted(fp2)])}' if fp2 else '')
        + f'\n\nRecall {r2*100:.1f}% berarti dari {len(ground_truth_q2)} dokumen relevan, '
        f'sistem menemukan {len(rr2)} dan melewatkan {len(fn2)} dokumen.'
        + (f'\nDokumen yang terlewat: {", ".join(["D"+str(d) for d in sorted(fn2)])}' if fn2 else '')
        + '\n\nPerlu diperhatikan bahwa query ini lebih menantang karena kata "vaksin", "anak", '
        'dan "balita" bisa bermakna luas dalam konteks komentar sosial media.',
        'green'
    )

    # --- C.4 Analisis Kualitas ---
    pdf.add_page()
    pdf.sub_judul('C.4 Ringkasan dan Analisis Kualitas Keseluruhan')

    pdf.paragraf(
        'Tabel berikut merangkum performa sistem pada kedua skenario pengujian. '
        'Rata-rata dihitung dari kedua query untuk memberikan gambaran umum kualitas sistem.'
    )

    pdf.tabel(
        ['Metrik', f'Query 1', f'Query 2', 'Rata-rata'],
        [
            ['Precision', f'{p1:.4f} ({p1*100:.1f}%)', f'{p2:.4f} ({p2*100:.1f}%)', f'{(p1+p2)/2:.4f} ({(p1+p2)/2*100:.1f}%)'],
            ['Recall', f'{r1:.4f} ({r1*100:.1f}%)', f'{r2:.4f} ({r2*100:.1f}%)', f'{(r1+r2)/2:.4f} ({(r1+r2)/2*100:.1f}%)'],
            ['F-Measure', f'{f1:.4f} ({f1*100:.1f}%)', f'{f2:.4f} ({f2*100:.1f}%)', f'{(f1+f2)/2:.4f} ({(f1+f2)/2*100:.1f}%)'],
        ],
        [30, 50, 50, 60]
    )

    pdf.ln(3)

    avg_p = (p1+p2)/2
    avg_r = (r1+r2)/2
    avg_f = (f1+f2)/2

    quality = ""
    if avg_f >= 0.7:
        quality = "BAIK"
        quality_desc = "Sistem mampu menemukan dokumen relevan dengan akurasi yang tinggi."
    elif avg_f >= 0.4:
        quality = "CUKUP BAIK"
        quality_desc = "Sistem mampu menemukan sebagian besar dokumen relevan dengan cukup akurat."
    else:
        quality = "PERLU PERBAIKAN"
        quality_desc = "Sistem masih memiliki keterbatasan yang perlu diperbaiki."

    pdf.highlight_box(
        f'INTERPRETASI HASIL EVALUASI KESELURUHAN:\n\n'
        f'1. Precision Rata-rata: {avg_p:.4f} ({avg_p*100:.1f}%)\n'
        f'Artinya: Dari setiap 10 dokumen yang ditampilkan oleh sistem, sekitar '
        f'{avg_p*10:.1f} di antaranya benar-benar relevan. Sisanya adalah "noise" atau '
        f'dokumen yang kurang tepat. Precision yang baik penting agar pengguna tidak perlu '
        f'menyaring banyak hasil yang tidak berguna.\n\n'
        f'2. Recall Rata-rata: {avg_r:.4f} ({avg_r*100:.1f}%)\n'
        f'Artinya: Sistem berhasil menemukan sekitar {avg_r*100:.1f}% dari seluruh dokumen '
        f'yang relevan. Sisanya ({(1-avg_r)*100:.1f}%) terlewat dan tidak muncul di Top-10. '
        f'Recall penting untuk memastikan informasi penting tidak terlewatkan.\n\n'
        f'3. F-Measure Rata-rata: {avg_f:.4f} ({avg_f*100:.1f}%)\n'
        f'F-Measure menggabungkan Precision dan Recall menjadi satu angka. Menggunakan '
        f'harmonic mean (bukan rata-rata biasa) agar sistem yang hanya bagus di satu aspek '
        f'tetapi buruk di aspek lain tidak mendapat skor tinggi.\n\n'
        f'PENILAIAN KESELURUHAN: {quality}\n{quality_desc}',
        'green'
    )

    # ===================================================================
    # KESIMPULAN & SARAN
    # ===================================================================
    pdf.add_page()
    pdf.judul_bagian('Kesimpulan dan Saran')

    pdf.sub_judul('Kesimpulan')
    pdf.paragraf(
        'Berdasarkan seluruh analisis yang telah dilakukan dalam laporan ini, berikut adalah '
        'kesimpulan utama dari implementasi dan evaluasi Mini Search Engine:'
    )

    pdf.highlight_box(
        'KESIMPULAN UTAMA:\n\n'
        '1. TENTANG IDF (Bobot Kata):\n'
        f'Analisis IDF membuktikan bahwa kata yang jarang muncul memiliki bobot yang lebih '
        f'tinggi. Dalam dataset kita, kata "kejang" (IDF={idf2:.4f}) jauh lebih informatif '
        f'dibandingkan "imunisasi" (IDF={idf1:.4f}). Ini sesuai dengan prinsip dasar Information '
        f'Retrieval bahwa kata yang memiliki daya pembeda tinggi lebih berharga untuk pencarian. '
        f'Prinsip ini juga digunakan oleh mesin pencari besar seperti Google.\n\n'
        '2. TENTANG NORMALISASI:\n'
        'Perbandingan antara Cosine Similarity dan Dot Product menunjukkan bahwa normalisasi '
        'sangat penting untuk menghasilkan peringkat yang adil. Tanpa normalisasi, dokumen '
        'panjang mendapat keuntungan tidak adil, sementara dokumen pendek yang sangat relevan '
        'bisa tergeser. Cosine Similarity mengatasi masalah ini dengan mengukur kemiripan '
        'berdasarkan arah vektor, bukan besarnya.\n\n'
        '3. TENTANG KUALITAS SISTEM:\n'
        f'Evaluasi dengan dua skenario query menunjukkan bahwa sistem memiliki kualitas yang '
        f'{quality.lower()} dengan F-Measure rata-rata {avg_f:.4f} ({avg_f*100:.1f}%). '
        f'Precision rata-rata {avg_p*100:.1f}% menunjukkan bahwa sebagian besar hasil pencarian '
        f'memang relevan, dan Recall rata-rata {avg_r*100:.1f}% menunjukkan bahwa sistem '
        f'berhasil menemukan sebagian dokumen relevan yang ada.',
        'green'
    )

    pdf.sub_judul('Kelebihan Sistem')
    pdf.paragraf(
        '1. Menggunakan Sastrawi Stemmer yang dirancang khusus untuk Bahasa Indonesia, '
        'sehingga proses stemming lebih akurat dibanding stemmer generik.\n\n'
        '2. Menerapkan Log Frequency Weighting untuk TF, yang mencegah dokumen dengan '
        'pengulangan kata yang berlebihan mendominasi hasil pencarian.\n\n'
        '3. Cosine Similarity memastikan peringkat yang adil terlepas dari panjang dokumen.\n\n'
        f'4. Sistem berhasil memproses {N} dokumen dengan bahasa informal/slang media sosial '
        'yang merupakan tantangan tersendiri dalam NLP Bahasa Indonesia.'
    )

    pdf.sub_judul('Keterbatasan dan Saran Perbaikan')
    pdf.paragraf(
        '1. KETERBATASAN BAHASA SLANG: Banyak kata singkatan dan bahasa gaul (seperti "gk", '
        '"bgt", "tp") yang mungkin tidak tertangani dengan baik oleh stemmer Sastrawi. '
        'Saran: Menambahkan kamus normalisasi slang sebelum proses stemming.\n\n'
        '2. UKURAN KOLEKSI: Dengan hanya 50 dokumen, koleksi ini masih relatif kecil. '
        'Pada koleksi yang lebih besar, performa TF-IDF biasanya meningkat karena statistik '
        'yang lebih representatif. Saran: Menguji dengan dataset yang lebih besar.\n\n'
        '3. SINONIM DAN TYPO: Sistem belum bisa menangani sinonim (misal "vaksin" dan "imunisasi" '
        'yang bermakna mirip) atau kesalahan ketik ("imunasi" vs "imunisasi"). '
        'Saran: Menggunakan teknik query expansion atau word embedding.\n\n'
        '4. GROUND TRUTH SUBJEKTIF: Penentuan dokumen relevan dilakukan secara manual dan bisa '
        'berbeda jika dinilai oleh orang yang berbeda. Saran: Menggunakan beberapa penilai '
        'dan menghitung inter-annotator agreement.\n\n'
        '5. KONTEKS KALIMAT: Sistem berbasis bag-of-words tidak memahami konteks. Misalnya, '
        '"tidak demam setelah imunisasi" dan "demam setelah imunisasi" dianggap mirip karena '
        'mengandung kata yang sama. Saran: Mempertimbangkan teknik berbasis posisi (phrase query) '
        'atau pendekatan deep learning.'
    )

    pdf.highlight_box(
        'PENUTUP:\n\n'
        'Mini Search Engine ini berhasil mengimplementasikan konsep-konsep dasar Information '
        'Retrieval mulai dari pre-processing hingga ranked retrieval menggunakan Vector Space '
        'Model. Meskipun masih memiliki keterbatasan, sistem ini sudah mampu mendemonstrasikan '
        'bagaimana teori-teori yang dipelajari di perkuliahan Temu Kembali Informasi dapat '
        'diterapkan dalam aplikasi nyata yang bermanfaat.\n\n'
        'Semoga laporan ini dapat memberikan pemahaman yang jelas tentang cara kerja sistem '
        'temu kembali informasi dan bagaimana setiap komponen (pre-processing, indexing, '
        'TF-IDF weighting, dan cosine similarity) saling berkaitan untuk menghasilkan '
        'pencarian yang efektif.',
        'blue'
    )

    # ===== SAVE PDF =====
    output_path = os.path.join(SCRIPT_DIR, "Laporan_Analisis_Mini_Search_Engine.pdf")
    pdf.output(output_path)
    print(f"\n{'='*60}")
    print(f"  LAPORAN PDF BERHASIL DIBUAT!")
    print(f"  File: {output_path}")
    print(f"{'='*60}")


# ############################################################################
#
#   BAGIAN 4: MODE WEB - STREAMLIT UI
#   (Asal file: app.py)
#
#   Antarmuka web interaktif menggunakan Streamlit.
#   Menampilkan pre-processing, inverted index, TF-IDF, dan ranked retrieval.
#
#   Cara menjalankan: streamlit run semua_kode.py -- --mode web
#
# ############################################################################

def run_web():
    """Menjalankan Mini Search Engine dengan Streamlit Web UI."""
    import streamlit as st
    import pandas as pd

    st.set_page_config(
        page_title="Mini Search Engine",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Custom CSS ---
    st.markdown("""
    <style>
        /* ===== Global ===== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        .stApp {
            font-family: 'Inter', sans-serif;
        }

        /* ===== Hero Header ===== */
        .hero {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            padding: 2.5rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 20px 60px rgba(48, 43, 99, 0.4);
            position: relative;
            overflow: hidden;
        }
        .hero::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(139, 92, 246, 0.1) 0%, transparent 70%);
            animation: pulse 4s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 1; }
        }
        .hero h1 {
            color: #ffffff;
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0;
            position: relative;
            letter-spacing: -0.5px;
        }
        .hero .subtitle {
            color: #a78bfa;
            font-size: 1rem;
            margin-top: 0.5rem;
            font-weight: 400;
            position: relative;
        }
        .hero .badge-row {
            margin-top: 1rem;
            display: flex;
            justify-content: center;
            gap: 0.7rem;
            flex-wrap: wrap;
            position: relative;
        }
        .hero .badge {
            background: rgba(139, 92, 246, 0.2);
            border: 1px solid rgba(139, 92, 246, 0.4);
            color: #c4b5fd;
            padding: 0.3rem 0.9rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 500;
            backdrop-filter: blur(10px);
        }

        /* ===== Stat Cards ===== */
        .stat-card {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
            border: 1px solid rgba(139, 92, 246, 0.3);
            padding: 1.3rem;
            border-radius: 16px;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(139, 92, 246, 0.2);
        }
        .stat-card .stat-value {
            font-size: 2rem;
            font-weight: 800;
            color: #a78bfa;
            line-height: 1;
        }
        .stat-card .stat-label {
            color: #94a3b8;
            font-size: 0.82rem;
            margin-top: 0.4rem;
            font-weight: 500;
        }

        /* ===== Result Card ===== */
        .result-card {
            background: linear-gradient(135deg, #1e1b4b 0%, #1e293b 100%);
            border: 1px solid rgba(139, 92, 246, 0.2);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
            transition: transform 0.2s, border-color 0.3s, box-shadow 0.3s;
        }
        .result-card:hover {
            transform: translateY(-2px);
            border-color: rgba(139, 92, 246, 0.5);
            box-shadow: 0 8px 30px rgba(139, 92, 246, 0.15);
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.7rem;
        }
        .rank-badge {
            background: linear-gradient(135deg, #7c3aed, #a78bfa);
            color: white;
            font-weight: 700;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.85rem;
        }
        .score-badge {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.3);
            color: #4ade80;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .doc-label {
            color: #94a3b8;
            font-size: 0.78rem;
            font-weight: 500;
        }
        .doc-text {
            color: #e2e8f0;
            font-size: 0.92rem;
            line-height: 1.6;
            margin-top: 0.3rem;
        }
        .source-tag {
            display: inline-block;
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid rgba(59, 130, 246, 0.3);
            color: #60a5fa;
            padding: 0.15rem 0.6rem;
            border-radius: 8px;
            font-size: 0.72rem;
            margin-top: 0.6rem;
            font-weight: 500;
        }

        /* ===== Step Card (preprocessing) ===== */
        .step-card {
            background: linear-gradient(135deg, #1e1b4b 0%, #1e293b 100%);
            border: 1px solid rgba(139, 92, 246, 0.15);
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.8rem;
        }
        .step-number {
            display: inline-block;
            background: linear-gradient(135deg, #7c3aed, #a78bfa);
            color: white;
            font-weight: 700;
            width: 28px;
            height: 28px;
            line-height: 28px;
            text-align: center;
            border-radius: 50%;
            font-size: 0.82rem;
            margin-right: 0.6rem;
        }
        .step-title {
            color: #c4b5fd;
            font-weight: 600;
            font-size: 0.9rem;
            display: inline;
        }
        .step-content {
            color: #cbd5e1;
            font-size: 0.85rem;
            margin-top: 0.5rem;
            padding: 0.6rem 0.8rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            word-wrap: break-word;
            line-height: 1.5;
            max-height: 120px;
            overflow-y: auto;
        }

        /* ===== Section Title ===== */
        .section-title {
            color: #e2e8f0;
            font-size: 1.3rem;
            font-weight: 700;
            margin: 1.5rem 0 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .section-title .icon {
            font-size: 1.4rem;
        }

        /* ===== Info Box ===== */
        .info-box {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            color: #93c5fd;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        /* ===== Search Box Styling ===== */
        .search-container {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 20px;
            padding: 1.8rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 40px rgba(139, 92, 246, 0.15);
        }

        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #1e1b4b; }
        ::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 3px; }

        /* Token tags */
        .token-tag {
            display: inline-block;
            background: rgba(139, 92, 246, 0.2);
            border: 1px solid rgba(139, 92, 246, 0.3);
            color: #c4b5fd;
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            font-size: 0.78rem;
            margin: 0.15rem;
            font-family: 'Courier New', monospace;
        }

        /* Table styling */
        .dataframe {
            font-size: 0.82rem !important;
        }

        /* Sidebar title */
        .sidebar-title {
            color: #a78bfa;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # ============================================================
    # Initialize Engine (cached)
    # ============================================================
    @st.cache_resource
    def init_engine():
        eng = MiniSearchEngine()
        if os.path.exists(CSV_FILE):
            eng.load_documents_from_csv(CSV_FILE)
            eng.build_index()
            eng.compute_tfidf()
        return eng

    engine = init_engine()

    # ============================================================
    # Hero Header
    # ============================================================
    st.markdown("""
    <div class="hero">
        <h1>🔍 Mini Search Engine</h1>
        <div class="subtitle">Mesin Pencari Mini — Temu Kembali Informasi</div>
        <div class="badge-row">
            <span class="badge">📝 Pre-processing</span>
            <span class="badge">📊 TF-IDF Weighting</span>
            <span class="badge">📐 Cosine Similarity</span>
            <span class="badge">🇮🇩 Sastrawi Stemmer</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # Stat Cards
    # ============================================================
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{engine.num_docs}</div>
            <div class="stat-label">📄 Total Dokumen</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(engine.vocabulary)}</div>
            <div class="stat-label">🔤 Term Unik</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(engine.inverted_index)}</div>
            <div class="stat-label">📑 Index Entries</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{"✅" if SASTRAWI_AVAILABLE else "❌"}</div>
            <div class="stat-label">🧠 Sastrawi Stemmer</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # Sidebar - Navigation
    # ============================================================
    with st.sidebar:
        st.markdown('<div class="sidebar-title">📌 Navigasi</div>', unsafe_allow_html=True)
        page = st.radio(
            "Pilih halaman:",
            ["🔍 Pencarian", "📝 Pre-processing", "📑 Inverted Index", "📊 TF-IDF Matrix", "📄 Koleksi Dokumen"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown('<div class="sidebar-title">ℹ️ Tentang</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>Mini Search Engine</strong><br>
            Program ini mengimplementasikan:<br>
            • Text Pre-processing<br>
            • Inverted Index<br>
            • TF-IDF (Log Freq. Weighting)<br>
            • Vector Space Model<br>
            • Cosine Similarity<br><br>
            <em>Mata Kuliah: Temu Kembali Informasi</em>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="sidebar-title">⚙️ Pengaturan</div>', unsafe_allow_html=True)
        top_k = st.slider("Jumlah hasil pencarian", 3, 20, 10)

    # ============================================================
    # Page: Pencarian
    # ============================================================
    if page == "🔍 Pencarian":
        st.markdown('<div class="section-title"><span class="icon">🔍</span> Pencarian Dokumen</div>', unsafe_allow_html=True)

        # Search box
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        query = st.text_input(
            "Masukkan query pencarian:",
            placeholder="Contoh: demam setelah imunisasi, vaksin anak, efek samping suntik...",
            label_visibility="visible"
        )
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        with col_btn1:
            search_clicked = st.button("🔍 Cari", type="primary", use_container_width=True)
        with col_btn2:
            clear_clicked = st.button("🗑️ Hapus", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if clear_clicked:
            st.rerun()

        # Quick search suggestions
        st.markdown("**💡 Contoh query:**")
        suggestions = ["demam setelah imunisasi", "vaksin anak balita", "efek samping suntik DPT", "jadwal imunisasi bayi", "kompres setelah vaksin"]
        cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            with cols[i]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state['auto_query'] = sug
                    st.rerun()

        # Use auto_query if set
        active_query = query
        if 'auto_query' in st.session_state and not query:
            active_query = st.session_state.pop('auto_query')

        if active_query and (search_clicked or active_query != query or active_query):
            results, query_tokens = engine.search(active_query, top_k=top_k)

            # Query info
            st.markdown(f"""
            <div class="info-box" style="margin: 1rem 0;">
                <strong>Query:</strong> "{active_query}"<br>
                <strong>Token setelah pre-processing:</strong> {', '.join(query_tokens) if query_tokens else '-'}<br>
                <strong>Term ditemukan di index:</strong> {', '.join([t for t in query_tokens if t in engine.inverted_index]) if query_tokens else '-'}<br>
                <strong>Jumlah hasil:</strong> {len(results)} dokumen
            </div>
            """, unsafe_allow_html=True)

            if results:
                for r in results:
                    rank = results.index(r) + 1
                    text = r['text']
                    display_text = text[:250] + "..." if len(text) > 250 else text
                    source_html = ""
                    if r['source'] and r['source'] != '-':
                        source_html = f'<div><span class="source-tag">🔗 {r["source"]}</span></div>'

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-header">
                            <div>
                                <span class="rank-badge">#{rank}</span>
                                <span class="doc-label" style="margin-left: 0.5rem;">Dokumen D{r['doc_id']}</span>
                            </div>
                            <span class="score-badge">Score: {r['score']:.6f}</span>
                        </div>
                        <div class="doc-text">{display_text}</div>
                        {source_html}
                    </div>
                    """, unsafe_allow_html=True)

                # Score chart
                st.markdown('<div class="section-title"><span class="icon">📊</span> Visualisasi Skor</div>', unsafe_allow_html=True)
                chart_data = pd.DataFrame({
                    'Dokumen': [f"D{r['doc_id']}" for r in results],
                    'Cosine Similarity': [r['score'] for r in results]
                })
                st.bar_chart(chart_data.set_index('Dokumen'), color='#7c3aed')
            else:
                st.markdown("""
                <div class="info-box" style="border-color: rgba(251, 146, 60, 0.3); color: #fdba74;">
                    ⚠️ Tidak ada dokumen relevan ditemukan. Coba kata kunci lain.
                </div>
                """, unsafe_allow_html=True)

    # ============================================================
    # Page: Pre-processing
    # ============================================================
    elif page == "📝 Pre-processing":
        st.markdown('<div class="section-title"><span class="icon">📝</span> Demo Pre-processing</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            Pipeline pre-processing terdiri dari 5 tahap:<br>
            <strong>1.</strong> Case Folding → <strong>2.</strong> Punctuation Removal → <strong>3.</strong> Tokenisasi → <strong>4.</strong> Stopword Removal → <strong>5.</strong> Stemming (Sastrawi)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Option to choose from documents or custom text
        input_mode = st.radio("Pilih input:", ["📄 Dari dokumen", "✏️ Teks custom"], horizontal=True)

        if input_mode == "📄 Dari dokumen":
            doc_ids = sorted(engine.documents.keys())
            selected_doc = st.selectbox("Pilih dokumen:", doc_ids, format_func=lambda x: f"D{x}: {engine.documents[x][:80]}...")
            input_text = engine.documents[selected_doc]
        else:
            input_text = st.text_area("Masukkan teks:", value="Saya ingin tahu tentang jadwal imunisasi anak balita", height=100)

        if input_text:
            steps = engine.preprocess_steps(input_text)

            step_data = [
                ("1", "Case Folding", "Mengubah semua huruf menjadi lowercase", steps['case_folding']),
                ("2", "Punctuation Removal", "Menghapus tanda baca, URL, angka, emoji", steps['punctuation_removal']),
                ("3", "Tokenisasi", "Memecah teks menjadi kata-kata (token)", steps['tokenization']),
                ("4", "Stopword Removal", "Menghapus kata-kata umum yang tidak bermakna", steps['stopword_removal']),
                ("5", "Stemming (Sastrawi)", "Mengubah kata ke bentuk dasarnya", steps['stemming']),
            ]

            for num, title, desc, content in step_data:
                if isinstance(content, list):
                    token_html = ' '.join([f'<span class="token-tag">{t}</span>' for t in content])
                    content_display = f"{token_html}<br><em style='color:#94a3b8;font-size:0.78rem;'>({len(content)} token)</em>"
                else:
                    content_display = content

                st.markdown(f"""
                <div class="step-card">
                    <span class="step-number">{num}</span>
                    <span class="step-title">{title}</span>
                    <div style="color:#94a3b8;font-size:0.75rem;margin-left:2.5rem;margin-top:0.1rem;">{desc}</div>
                    <div class="step-content">{content_display}</div>
                </div>
                """, unsafe_allow_html=True)

            # Summary
            orig_count = len(steps['tokenization'])
            final_count = len(steps['stemming'])
            reduction = orig_count - final_count
            pct = (reduction / orig_count * 100) if orig_count > 0 else 0

            st.markdown(f"""
            <div class="stat-card" style="margin-top:1rem;">
                <div style="display:flex;justify-content:space-around;text-align:center;">
                    <div>
                        <div class="stat-value">{orig_count}</div>
                        <div class="stat-label">Token Awal</div>
                    </div>
                    <div>
                        <div class="stat-value" style="color:#f87171;">-{reduction}</div>
                        <div class="stat-label">Dihapus</div>
                    </div>
                    <div>
                        <div class="stat-value" style="color:#4ade80;">{final_count}</div>
                        <div class="stat-label">Token Akhir</div>
                    </div>
                    <div>
                        <div class="stat-value">{pct:.0f}%</div>
                        <div class="stat-label">Reduksi</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ============================================================
    # Page: Inverted Index
    # ============================================================
    elif page == "📑 Inverted Index":
        st.markdown('<div class="section-title"><span class="icon">📑</span> Inverted Index</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            Inverted Index memetakan setiap <strong>term</strong> ke daftar dokumen (<strong>posting list</strong>) yang mengandungnya, beserta frekuensi kemunculannya (TF).
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Search filter
        filter_term = st.text_input("🔎 Filter term:", placeholder="Ketik untuk mencari term...")

        # Build dataframe
        index_data = []
        for term in sorted(engine.inverted_index.keys()):
            if filter_term and filter_term.lower() not in term.lower():
                continue
            postings = engine.inverted_index[term]
            df_val = len(postings)
            posting_str = ', '.join([f"D{did}:{tf}" for did, tf in sorted(postings.items())])
            index_data.append({
                'Term': term,
                'DF': df_val,
                'Posting List (DocID:TF)': posting_str
            })

        if index_data:
            df_index = pd.DataFrame(index_data)
            st.markdown(f"Menampilkan **{len(index_data)}** dari **{len(engine.vocabulary)}** term")
            st.dataframe(df_index, use_container_width=True, height=500)
        else:
            st.info("Tidak ada term yang cocok dengan filter.")

    # ============================================================
    # Page: TF-IDF Matrix
    # ============================================================
    elif page == "📊 TF-IDF Matrix":
        st.markdown('<div class="section-title"><span class="icon">📊</span> Matriks TF-IDF</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <strong>TF (Log Frequency):</strong> w(t,d) = 1 + log₁₀(tf) &nbsp;jika tf &gt; 0<br>
            <strong>IDF:</strong> idf(t) = log₁₀(N / df(t))<br>
            <strong>TF-IDF:</strong> w(t,d) = TF × IDF
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            max_terms = st.slider("Jumlah term:", 5, min(100, len(engine.vocabulary)), 20)
        with col_b:
            max_docs = st.slider("Jumlah dokumen:", 3, min(50, engine.num_docs), 10)

        sorted_terms = sorted(engine.vocabulary)[:max_terms]
        sorted_docs = sorted(engine.documents.keys())[:max_docs]

        matrix_data = []
        for term in sorted_terms:
            row = {'Term': term}
            for doc_id in sorted_docs:
                w = engine.tfidf_weights.get(term, {}).get(doc_id, 0)
                row[f'D{doc_id}'] = round(w, 4)
            matrix_data.append(row)

        df_matrix = pd.DataFrame(matrix_data)
        st.dataframe(
            df_matrix.style.background_gradient(cmap='Purples', subset=[f'D{d}' for d in sorted_docs]),
            use_container_width=True,
            height=500
        )

        # Show IDF values
        with st.expander("📐 Lihat Nilai IDF untuk setiap Term"):
            idf_data = []
            for term in sorted(engine.vocabulary):
                df_val = len(engine.inverted_index[term])
                idf = math.log10(engine.num_docs / df_val) if df_val > 0 else 0
                idf_data.append({'Term': term, 'DF': df_val, 'IDF': round(idf, 4)})
            df_idf = pd.DataFrame(idf_data)
            st.dataframe(df_idf, use_container_width=True, height=400)

    # ============================================================
    # Page: Koleksi Dokumen
    # ============================================================
    elif page == "📄 Koleksi Dokumen":
        st.markdown('<div class="section-title"><span class="icon">📄</span> Koleksi Dokumen</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            Total <strong>{engine.num_docs}</strong> dokumen dimuat dari file CSV.<br>
            Dataset: Komentar sosial media tentang imunisasi balita.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Build document table
        doc_table = []
        for doc_id in sorted(engine.documents.keys()):
            doc_table.append({
                'No': doc_id,
                'Komentar': engine.documents[doc_id],
                'Sumber': engine.doc_sources.get(doc_id, '-'),
                'Jumlah Token (Setelah Pre-processing)': len(engine.processed_docs.get(doc_id, []))
            })

        df_docs = pd.DataFrame(doc_table)
        st.dataframe(df_docs, use_container_width=True, height=600)

        # Expandable document details
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🔎 Detail Dokumen:**")
        selected = st.selectbox(
            "Pilih dokumen untuk melihat detail:",
            sorted(engine.documents.keys()),
            format_func=lambda x: f"D{x}: {engine.documents[x][:60]}..."
        )

        if selected:
            doc = engine.documents[selected]
            tokens = engine.processed_docs.get(selected, [])

            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">
                    <span class="rank-badge">D{selected}</span>
                    <span class="doc-label">{len(tokens)} token</span>
                </div>
                <div class="doc-text">{doc}</div>
                <div style="margin-top:0.8rem;">
                    <strong style="color:#a78bfa;font-size:0.85rem;">Token setelah pre-processing:</strong><br>
                    {''.join([f'<span class="token-tag">{t}</span>' for t in tokens])}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ############################################################################
#
#   BAGIAN 5: MAIN - MENU UTAMA
#
#   Otomatis mendeteksi apakah dijalankan lewat Streamlit atau terminal.
#   - Jika lewat Streamlit   : langsung masuk Web UI
#   - Jika lewat terminal    : pilih mode (cli/pdf/web)
#
#   Cara menjalankan Web UI cukup:
#       streamlit run semua_kode.py
#
# ############################################################################

def _is_running_with_streamlit():
    """Deteksi apakah script sedang dijalankan oleh Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

# Jika dijalankan lewat Streamlit, langsung masuk web UI
if _is_running_with_streamlit():
    run_web()
elif __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini Search Engine - Semua dalam Satu File")
    parser.add_argument('--mode', choices=['cli', 'pdf', 'web'],
                        help='Mode: cli (terminal), pdf (generate laporan), web (Streamlit UI)')
    args, _ = parser.parse_known_args()

    if args.mode == 'cli':
        run_cli()
    elif args.mode == 'pdf':
        run_pdf()
    elif args.mode == 'web':
        run_web()
    else:
        # Menu interaktif jika tidak ada argumen
        print()
        print("+" + "="*58 + "+")
        print("|" + " "*58 + "|")
        print("|" + "MINI SEARCH ENGINE - SEMUA KODE".center(58) + "|")
        print("|" + "Pilih mode yang ingin dijalankan".center(58) + "|")
        print("|" + " "*58 + "|")
        print("+" + "="*58 + "+")
        print()
        print("  [1] CLI  - Pencarian interaktif di terminal")
        print("  [2] PDF  - Generate laporan PDF")
        print("  [3] WEB  - Web UI dengan Streamlit")
        print()

        try:
            choice = input("  Pilih mode (1/2/3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Keluar.\n")
            sys.exit(0)

        if choice == '1':
            run_cli()
        elif choice == '2':
            run_pdf()
        elif choice == '3':
            print("\n  Membuka Web UI...")
            print("  Jalankan perintah: streamlit run semua_kode.py\n")
        else:
            print("\n  Pilihan tidak valid. Keluar.\n")

