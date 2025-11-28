import time
import math
import sqlite3
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from pydantic import BaseModel

import mmh3
from bitarray import bitarray


# =======================================================
# CONFIG
# =======================================================

CORPUS_PATH = "corpus"
DB_PATH = "passwords.db"

BF_SIZE = 20_000_000
BF_HASHES = 7

HASH_RATE = 1e10  # For your time_to_crack() formula


# =======================================================
# BLOOM FILTER
# =======================================================

class BloomFilter:
    def __init__(self, size=BF_SIZE, hashes=BF_HASHES):
        self.size = size
        self.hashes = hashes
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def _hashes(self, item):
        for seed in range(self.hashes):
            yield mmh3.hash(item, seed) % self.size

    def add(self, item):
        for h in self._hashes(item):
            self.bit_array[h] = 1

    def check(self, item):
        return all(self.bit_array[h] for h in self._hashes(item))


bloom = BloomFilter()


# =======================================================
# DATABASE
# =======================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS password_dict (
            password TEXT PRIMARY KEY,
            freq INTEGER
        )
    """)

    conn.commit()
    conn.close()


def load_corpus_into_db(folder: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    folder_path = Path(folder)
    if not folder_path.exists():
        print("Corpus folder not found.")
        return

    freq_map = Counter()

    for file in folder_path.glob("*.txt"):
        with file.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                pw = line.strip()
                if pw:
                    freq_map[pw] += 1

    data = [(pw, freq) for pw, freq in freq_map.items()]
    c.executemany("""
        INSERT OR REPLACE INTO password_dict (password, freq)
        VALUES (?, ?)
    """, data)

    conn.commit()
    conn.close()

    for pw in freq_map.keys():
        bloom.add(pw)

    print(f"Loaded {len(freq_map)} passwords into DB and Bloom filter.")


def get_freq_bounds():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM password_dict")
    count = c.fetchone()[0]

    if count == 0:
        conn.close()
        return 1, 1

    c.execute("SELECT MIN(freq), MAX(freq) FROM password_dict")
    mn, mx = c.fetchone()

    conn.close()
    return mn, mx


# =======================================================
# YOUR EXACT SCORING FUNCTIONS (UNCHANGED)
# =======================================================

def entropy_bits(password: str) -> float:
    m = len(password)

    lower = any(c.islower() for c in password)
    upper = any(c.isupper() for c in password)
    digits = any(c.isdigit() for c in password)
    special = any(not c.isalnum() for c in password)

    N = (
        (26 if lower else 0) +
        (26 if upper else 0) +
        (10 if digits else 0) +
        (32 if special else 0)
    )
    if N == 0:
        return 0.0

    return m * math.log2(N)


def entropy_based(password: str) -> float:
    if not password:
        return 0.0
    H = entropy_bits(password)
    score = 100 * (H / 112)
    return max(0, min(round(score), 100))


def rule_based(password: str) -> float:
    freq_table = Counter()

    l = len(password)

    sets_used = sum([
        any(c.islower() for c in password),
        any(c.isupper() for c in password),
        any(c.isdigit() for c in password),
        any(not c.isalnum() for c in password),
    ])

    n = sets_used

    k = 0
    last_type = None
    for c in password:
        t = (
            "L" if c.islower() else
            "U" if c.isupper() else
            "D" if c.isdigit() else
            "S"
        )
        if t == last_type:
            k += 1
        last_type = t

    s = 1 if any(not c.isalnum() for c in password) else 0

    specials = [i for i, c in enumerate(password) if not c.isalnum()]
    p = 1 if specials and (all(i < 2 for i in specials) or all(i >= l - 2 for i in specials)) else 0

    d = 0
    for i in range(l):
        for j in range(i + 3, l + 1):
            if password[i:j].lower() in freq_table:
                d += 1

    S_raw = n + (k / l) + s - p - (d / l)
    S_rule = 10 * S_raw

    S_rule = max(0, min(round(S_rule), 100))
    return S_rule


def hybrid_score(dictionary: float, rule: float, entropy: float) -> float:
    return round(dictionary * 0.3 + rule * 0.3 + entropy * 0.4, 2)


def fos_check(rule_s: float, dict_s: float) -> bool:
    return rule_s >= 70 and dict_s <= 30


def usability_score(password: str, rule_s: float) -> float:
    length = len(password)
    U = 50 * (length / 20) + 50 * (1 - rule_s / 100)
    return max(0, min(round(U), 100))


def usability_rating(score: float) -> str:
    if score < 20:
        return "Unusable"
    if score < 40:
        return "Poor"
    if score < 60:
        return "Acceptable"
    if score < 80:
        return "Good"
    return "Excellent"


def time_to_crack(password: str) -> float:
    H = entropy_bits(password)
    N_guesses = 2 ** H
    T = N_guesses / HASH_RATE
    return T


# =======================================================
# DICTIONARY SCORING (DYNAMIC 0–100)
# =======================================================

def dictionary_based(pwd: str) -> float:
    # Fast Bloom check
    if not bloom.check(pwd):
        return 100.0

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT freq FROM password_dict WHERE password = ?", (pwd,))
    row = c.fetchone()
    conn.close()

    if not row:
        return 100.0

    freq = row[0]
    mn, mx = get_freq_bounds()

    if mn == mx:
        return 0.0

    score = 100 * (1 - ((freq - mn) / (mx - mn)))
    return round(max(0, min(score, 100)), 2)


# =======================================================
# FASTAPI SERVICE
# =======================================================

class PasswordRequest(BaseModel):
    password: str


app = FastAPI()


@app.on_event("startup")
def startup():
    init_db()
    load_corpus_into_db(CORPUS_PATH)


@app.post("/evaluate-password")
def evaluate_password(req: PasswordRequest):
    pwd = req.password

    start = time.time()

    with ThreadPoolExecutor() as ex:
        f_entropy = ex.submit(entropy_based, pwd)
        f_rule = ex.submit(rule_based, pwd)
        f_dict = ex.submit(dictionary_based, pwd)

        entropy_s = f_entropy.result()
        rule_s = f_rule.result()
        dict_s = f_dict.result()

    hybrid_s = hybrid_score(dict_s, rule_s, entropy_s)
    fos = fos_check(rule_s, dict_s)

    usability_s = usability_score(pwd, rule_s)
    usability_r = usability_rating(usability_s)

    crack_seconds = time_to_crack(pwd)

    total_ms = (time.time() - start) * 1000

    return {
        "entropy_score": entropy_s,
        "rule_score": rule_s,
        "dictionary_score": dict_s,
        "hybrid_score": hybrid_s,
        "fos_check": fos,
        "usability_score": usability_s,
        "usability_rating": usability_r,
        "time_to_crack_seconds": crack_seconds,
        "total_time_ms": round(total_ms, 3)
    }
