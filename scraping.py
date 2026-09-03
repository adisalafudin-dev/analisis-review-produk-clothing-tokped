"""
Scraping Review Tokopedia
=========================
Memanfaatkan halaman /review yang DIIZINKAN oleh robots.txt:
  - Allow: /*/review
  - Allow: /*/*/review

Strategi:
  1. Fetch halaman /review via HTTP GET untuk mendapatkan cookies + product ID
  2. Panggil GraphQL API dengan query yang sama persis seperti yang dipakai
     oleh website Tokopedia sendiri, lengkap dengan pagination
  3. Fallback ke parsing HTML kalau GraphQL gagal

Cara pakai:
  # Scrape satu produk (URL langsung):
    python scraping.py "https://www.tokopedia.com/erigo/erigo-chino-pants-..."

  # Scrape semua produk dari produk.py (batch mode):
    python scraping.py

Install dulu:
    pip install requests beautifulsoup4
"""

import sys
import re
import time
import csv
import random
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

# ============ KONFIGURASI ============
TARGET_PER_PRODUK = 1000         # maks review yang diambil per produk
REVIEWS_PER_PAGE = 10            # review per halaman dari API
JEDA_HALAMAN = (1, 3)            # jeda acak (detik) antar halaman review
JEDA_ANTAR_PRODUK = (4, 7)       # jeda acak (detik) antar produk
OUTPUT_FILE = "reviews_tokopedia.csv"
DEBUG_DIR = Path("debug")        # folder untuk debug files
MAX_HALAMAN = 20                 # maks halaman review per produk
TIMEOUT = 20                     # timeout per request (detik)
# =====================================

# GraphQL query -- diambil dari JS bundle resmi Tokopedia
# (chunk.review-common-view.*.esm.js)
GRAPHQL_QUERY = (
    "query productReviewList($productID:String!,$page:Int!,$limit:Int!,"
    "$sortBy:String,$filterBy:String){productrevGetProductReviewList("
    " productID:$productID,page:$page,limit:$limit,sortBy:$sortBy,"
    "filterBy:$filterBy,){productID list{id:feedbackID variantName "
    "message productRating reviewCreateTime reviewCreateTimestamp "
    "isReportable isAnonymous imageAttachments{attachmentID "
    "imageThumbnailUrl imageUrl}videoAttachments{attachmentID videoUrl}"
    "reviewResponse{message createTime}user{userID fullName image url}"
    "likeDislike{totalLike likeStatus}stats{key formatted count}"
    "badRatingReasonFmt}shop{shopID name url image}hasNext totalReviews}}"
)

GRAPHQL_URL = "https://gql.tokopedia.com/graphql/productReviewList"


def jeda(rentang):
    """Tidur selama durasi acak dalam rentang (min, max) detik."""
    time.sleep(random.uniform(*rentang))


def bersihkan_url(url):
    """Hapus query params dan tambahkan /review."""
    parsed = urlparse(url)
    url_bersih = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    url_bersih = url_bersih.rstrip("/")
    if not url_bersih.endswith("/review"):
        url_bersih += "/review"
    return url_bersih


def buat_session():
    """Buat requests session dengan headers realistis."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
    return session


def dapatkan_product_id(session, review_url):
    """Fetch halaman review untuk mendapatkan cookies dan product ID asli."""
    print(f"  [>] Membuka: {review_url}")

    resp = session.get(review_url, timeout=TIMEOUT)
    if resp.status_code != 200:
        print(f"  [!] Status {resp.status_code}")
        return None, resp.text

    html = resp.text

    # Cari product ID dari Apollo cache di HTML
    # Pattern: productrevProductDetailMPI{PRODUCT_ID}
    pid_match = re.search(r'productrevProductDetailMPI(\d+)', html)
    if pid_match:
        product_id = pid_match.group(1)
        print(f"  [info] Product ID: {product_id}")
        return product_id, html

    # Fallback: cari dari JSON
    pid_match = re.search(r'"productID"\s*:\s*"(\d+)"', html)
    if pid_match:
        product_id = pid_match.group(1)
        print(f"  [info] Product ID (fallback): {product_id}")
        return product_id, html

    print("  [!] Product ID tidak ditemukan")
    return None, html


def ambil_review_via_graphql(session, product_id, review_url, page=1):
    """Panggil GraphQL API untuk mengambil review di halaman tertentu."""
    gql_headers = {
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Origin": "https://www.tokopedia.com",
        "Referer": review_url,
        "X-Source": "tokopedia-lite",
        "X-Device": "desktop",
        "X-Tkpd-Lite-Service": "zeus",
    }

    payload = [{
        "operationName": "productReviewList",
        "variables": {
            "productID": product_id,
            "page": page,
            "limit": REVIEWS_PER_PAGE,
            "sortBy": "informative_score desc",
            "filterBy": "",
        },
        "query": GRAPHQL_QUERY,
    }]

    resp = session.post(GRAPHQL_URL, json=payload, headers=gql_headers, timeout=TIMEOUT)

    if resp.status_code != 200:
        return None, False, 0

    data = resp.json()

    if isinstance(data, list) and len(data) > 0:
        item = data[0]
    else:
        item = data

    if "errors" in item and item["errors"]:
        error_msg = item["errors"][0].get("message", "unknown")
        print(f"  [!] GraphQL error: {error_msg}")
        return None, False, 0

    review_data = (item.get("data") or {}).get("productrevGetProductReviewList", {})
    if not review_data:
        return None, False, 0

    reviews = review_data.get("list", [])
    has_next = review_data.get("hasNext", False)
    total = review_data.get("totalReviews", 0)

    hasil = []
    for r in reviews:
        message = (r.get("message") or "").strip()
        if not message or len(message) < 3:
            continue

        user = r.get("user") or {}
        reviewer_name = user.get("fullName", "")

        hasil.append({
            "text": message,
            "rating": r.get("productRating"),
            "reviewer": reviewer_name if not r.get("isAnonymous") else "Anonim",
            "variant": r.get("variantName", ""),
        })

    return hasil, has_next, total


def ekstrak_review_dari_html(html_content):
    """Fallback: ekstrak review dari HTML SSR pakai BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")
    reviews = []

    review_elements = soup.find_all(attrs={"data-testid": "lblItemUlasan"})
    for el in review_elements:
        teks = el.get_text(strip=True)
        if teks and len(teks) > 2:
            reviews.append({
                "text": teks,
                "rating": None,
                "reviewer": None,
                "variant": None,
            })

    # Coba tambahkan rating dari Apollo cache
    if reviews:
        scripts = soup.find_all("script")
        for script in scripts:
            script_text = script.string or ""
            if "productRating" in script_text:
                ratings = re.findall(r'"productRating"\s*:\s*(\d+)', script_text)
                for i, r in enumerate(ratings):
                    if i < len(reviews):
                        reviews[i]["rating"] = int(r)
                break

    return reviews


def ambil_review_produk(session, url_review, brand, target):
    """Ambil review dari satu produk, menggunakan GraphQL API + HTML fallback."""
    semua_review = []

    # Step 1: Fetch halaman review untuk cookies + product ID
    product_id, html = dapatkan_product_id(session, url_review)

    if not product_id:
        # Fallback: coba parse review dari HTML langsung
        print("  [!] Tidak ada product ID, coba fallback HTML...")
        reviews = ekstrak_review_dari_html(html)
        for r in reviews:
            r["brand"] = brand
        return reviews[:target]

    # Step 2: Ambil review via GraphQL API dengan pagination
    print(f"  [>] Mengambil review via GraphQL API...")

    for halaman in range(1, MAX_HALAMAN + 1):
        try:
            reviews, has_next, total = ambil_review_via_graphql(
                session, product_id, url_review, page=halaman
            )
        except Exception as e:
            print(f"  [!] Error di halaman {halaman}: {e}")
            break

        if reviews is None:
            # GraphQL gagal, fallback ke HTML (hanya halaman 1)
            if halaman == 1:
                print("  [!] GraphQL gagal, fallback ke HTML...")
                reviews = ekstrak_review_dari_html(html)
                for r in reviews:
                    r["brand"] = brand
                return reviews[:target]
            break

        if halaman == 1:
            print(f"  [info] Total review tersedia: {total}")

        # Deduplikasi
        teks_existing = {r["text"] for r in semua_review}
        baru = 0
        for r in reviews:
            if r["text"] not in teks_existing:
                r["brand"] = brand
                semua_review.append(r)
                teks_existing.add(r["text"])
                baru += 1

        print(f"  [page {halaman}] +{baru} review (total: {len(semua_review)}/{target})")

        # Cek target
        if len(semua_review) >= target:
            print(f"  [OK] Target {target} tercapai!")
            break

        # Cek apakah masih ada halaman berikutnya
        if not has_next:
            print(f"  [info] Sudah halaman terakhir.")
            break

        # Cek stuck (tidak ada review baru)
        if baru == 0:
            print(f"  [info] Tidak ada review baru, selesai.")
            break

        # Jeda antar halaman
        jeda(JEDA_HALAMAN)

    return semua_review[:target]


def simpan_csv(reviews, filepath):
    """Simpan review ke file CSV."""
    fieldnames = ["text", "rating", "brand", "reviewer", "variant"]
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reviews)
    print(f"\n[SAVED] {len(reviews)} review -> '{filepath}'")


def main():
    # --- Parse argumen CLI ---
    url_tunggal = None
    if len(sys.argv) > 1:
        url_tunggal = sys.argv[1]
        print(f"[MODE] URL tunggal: {url_tunggal}")

    # Tentukan daftar produk
    if url_tunggal:
        parsed = urlparse(url_tunggal)
        path_parts = [p for p in parsed.path.split("/") if p]
        brand = path_parts[0] if path_parts else "unknown"
        daftar_scrape = [(brand, url_tunggal)]
    else:
        try:
            from produk import PRODUK
        except ImportError:
            print("[ERROR] File produk.py tidak ditemukan.")
            print('   Gunakan: python scraping.py "URL_PRODUK"')
            sys.exit(1)

        daftar_scrape = []
        for brand, urls in PRODUK.items():
            for url in urls:
                daftar_scrape.append((brand, url))

        print(f"[MODE] Batch: {len(daftar_scrape)} produk dari {len(PRODUK)} brand")

    if not daftar_scrape:
        print("[ERROR] Tidak ada produk untuk di-scrape.")
        sys.exit(1)

    # --- Mulai scraping ---
    session = buat_session()
    semua_review = []

    for idx, (brand, product_url) in enumerate(daftar_scrape):
        url_review = bersihkan_url(product_url)
        print(f"\n{'='*60}")
        print(f"  Brand: {brand} ({idx+1}/{len(daftar_scrape)})")
        print(f"{'='*60}")

        try:
            reviews = ambil_review_produk(
                session=session,
                url_review=url_review,
                brand=brand,
                target=TARGET_PER_PRODUK,
            )
            semua_review.extend(reviews)
            print(f"  [total] Kumulatif: {len(semua_review)} review")

        except Exception as e:
            print(f"  [ERROR] {e}")

        if idx < len(daftar_scrape) - 1:
            print(f"  [wait] Jeda antar produk...")
            jeda(JEDA_ANTAR_PRODUK)

    # --- Simpan hasil ---
    simpan_csv(semua_review, OUTPUT_FILE)

    # --- Ringkasan ---
    print(f"\n{'='*60}")
    print(f"  RINGKASAN")
    print(f"{'='*60}")
    brands = {}
    for r in semua_review:
        brands[r["brand"]] = brands.get(r["brand"], 0) + 1
    for brand, count in brands.items():
        print(f"  {brand}: {count} review")
    print(f"  TOTAL: {len(semua_review)} review")


if __name__ == "__main__":
    main()