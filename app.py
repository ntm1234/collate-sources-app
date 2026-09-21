#!/usr/bin/env python3
"""
Improved Streamlit app: collate CSV/XLSX, dedupe by Title, TF-IDF keyphrases (prefer multiword),
and a robust stopword-aware fallback so Keywords are meaningful.
"""
import streamlit as st
import pandas as pd
import io
import numpy as np
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from collections import Counter

st.set_page_config(page_title="Collate Sources — Improved Keyphrases", layout="wide")
st.title("Collate CSV / Excel files — Improved TF‑IDF keyphrases, dedupe by Title")

# Sidebar controls
st.sidebar.header("TF‑IDF options")
ngram_min = st.sidebar.number_input("n‑gram min", min_value=1, max_value=3, value=1, step=1)
ngram_max = st.sidebar.number_input("n‑gram max", min_value=1, max_value=4, value=2, step=1)
if ngram_max < ngram_min:
    st.sidebar.error("n‑gram max must be >= n‑gram min")
top_n = st.sidebar.number_input("Keywords per record", min_value=1, max_value=10, value=3, step=1)
extra_sw_text = st.sidebar.text_area("Extra stopwords (comma-separated)", value="music, national, identity")
extra_stopwords = [w.strip().lower() for w in extra_sw_text.split(",") if w.strip()]

# Column candidate mapping and Excel letter fallbacks
desired_columns_candidates = {
    "Authors": ["Authors", "Author"],
    "Title": ["Title"],
    "Year": ["Year"],
    "Source": ["Source", "Journal"],
    "ArticleURL": ["ArticleURL", "URL"],
    "Type": ["Type"],
    "DOI": ["DOI"],
    "Abstract": ["Abstract", "Summary"],
    "FullTextURL": ["FullTextURL", "Fulltext"]
}
letter_to_index = {chr(ord('A')+i): i for i in range(26)}
excel_letters = {"Authors":'B', "Title":'C', "Year":'D', "Source":'E', "ArticleURL":'G', "Type":'K', "DOI":'L', "Abstract":'X', "FullTextURL":'Y'}

def find_column(df, candidates, fallback_letter=None):
    cols_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_map:
            return cols_map[cand.lower()]
    for c in df.columns:
        for cand in candidates:
            if cand.lower() in c.lower():
                return c
    if fallback_letter:
        idx = letter_to_index.get(fallback_letter.upper())
        if idx is not None and idx < len(df.columns):
            return df.columns[idx]
    return None

def read_uploaded_file(uploaded):
    name = uploaded.name.lower()
    raw = uploaded.read()
    if name.endswith('.xlsx'):
        try:
            return pd.read_excel(io.BytesIO(raw), engine='openpyxl')
        except Exception as e:
            st.error(f"Could not read Excel file {uploaded.name}: {e}")
            return None
    else:
        for enc in ('utf-8', 'latin1'):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc, engine='python')
            except Exception:
                continue
        st.error(f"Could not read CSV file {uploaded.name} (tried utf-8 and latin1).")
        return None

def clean_and_select(df):
    out = pd.DataFrame()
    for logical_name, candidates in desired_columns_candidates.items():
        col = find_column(df, candidates, fallback_letter=excel_letters.get(logical_name))
        if col is not None:
            out[logical_name] = df[col].astype(str).where(~df[col].isna(), "")
        else:
            out[logical_name] = ""
    out = out[list(desired_columns_candidates.keys())]
    return out

# Utility: build stopset (sklearn english + user extras + some extra tokens)
def build_stopset(extra_stopwords):
    extras = set(w.strip().lower() for w in (extra_stopwords or []) if w.strip())
    baseline = set(["music","national","identity"])  # keep if you want them excluded by default
    stopset = set(ENGLISH_STOP_WORDS) | extras | baseline
    return stopset

# TF-IDF keyphrase extraction with multiword boost
def tfidf_keyphrases(corpus_series, top_n=3, ngram_range=(1,2), extra_stopwords=None):
    corpus = corpus_series.fillna("").astype(str).tolist()
    if all(not s.strip() for s in corpus):
        return [""] * len(corpus)
    stopset = build_stopset(extra_stopwords)
    # token_pattern: alphabetic tokens length >= 3 only (reduces "and", "the", "in")
    vectorizer = TfidfVectorizer(stop_words=stopset, ngram_range=ngram_range, token_pattern=r'(?u)\b[a-zA-Z]{3,}\b', max_df=0.95, min_df=1)
    try:
        X = vectorizer.fit_transform(corpus)
    except Exception:
        return [""] * len(corpus)
    vocab = np.array(vectorizer.get_feature_names_out())
    if len(vocab) == 0:
        return [""] * len(corpus)
    keywords = []
    for i in range(X.shape[0]):
        row = X.getrow(i).toarray().ravel()
        if row.sum() == 0:
            keywords.append("")
            continue
        # compute boost for multiword phrases (e.g. bigrams/trigrams)
        idxs = np.where(row > 0)[0]
        # compute adjusted score = score * (1 + 0.25*(num_words-1))
        terms = []
        for idx in idxs:
            term = vocab[idx]
            score = row[idx]
            nwords = term.count(" ") + 1
            adj = score * (1.0 + 0.25 * (nwords - 1))
            terms.append((term, adj, score))
        # sort by adjusted score
        terms_sorted = sorted(terms, key=lambda x: x[1], reverse=True)
        chosen = []
        for term, adj, rawscore in terms_sorted:
            # filter out tokens containing digits or urls
            if re.search(r'\d', term): 
                continue
            if "http" in term or "www." in term or "@" in term:
                continue
            chosen.append(term)
            if len(chosen) >= top_n:
                break
        keywords.append(", ".join(chosen))
    return keywords

# Fallback: stopword-aware frequency extraction
def fallback_frequency(corpus_series, top_n=3, extra_stopwords=None):
    stopset = build_stopset(extra_stopwords)
    def extract(text):
        if not isinstance(text, str) or not text.strip():
            return ""
        s = text.lower()
        s = re.sub(r'[' + re.escape(string.punctuation) + r']', ' ', s)
        tokens = re.findall(r'[a-zA-Z]{3,}', s)  # alphabetic only, min length 3
        tokens = [t for t in tokens if t not in stopset]
        if not tokens:
            return ""
        c = Counter(tokens)
        return ", ".join([w for w,_ in c.most_common(top_n)])
    return [extract(t) for t in corpus_series.fillna("").astype(str).tolist()]

# Main UI
uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    dfs = []
    for f in uploaded_files:
        df = read_uploaded_file(f)
        if df is None:
            continue
        cleaned = clean_and_select(df)
        dfs.append(cleaned)
    if not dfs:
        st.warning("No readable files uploaded.")
        st.stop()
    combined = pd.concat(dfs, ignore_index=True)

    st.write(f"Combined rows before deduplication: {len(combined)}")
    st.dataframe(combined.head(40))

    # Deduplicate by Title (case-insensitive trimmed)
    combined['__title_for_dedupe'] = combined['Title'].astype(str).str.strip().str.lower()
    before = len(combined)
    combined = combined.drop_duplicates(subset=['__title_for_dedupe'], keep='first').reset_index(drop=True)
    after = len(combined)
    st.success(f"Deduplication by Title applied: removed {before-after} rows; {after} rows remain.")
    combined.drop(columns=['__title_for_dedupe'], inplace=True, errors=False)

    # Build corpus (Title + Abstract)
    combined['__corpus'] = (combined['Title'].fillna("") + " " + combined['Abstract'].fillna("")).astype(str)

    # Show corpus sample
    st.markdown("Sample Title + Abstract (first 6 rows):")
    st.write(combined['__corpus'].head(6).apply(lambda s: (s or "")[:400]))

    # Run TF-IDF
    st.markdown("Generating keywords (TF‑IDF with n‑gram boost)...")
    ngram_range = (int(ngram_min), int(ngram_max))
    kws = tfidf_keyphrases(combined['__corpus'], top_n=int(top_n), ngram_range=ngram_range, extra_stopwords=extra_stopwords)

    # If TF-IDF produced nothing useful, fallback
    if all(not k for k in kws):
        st.info("TF‑IDF produced no keyphrases — using stopword-aware frequency fallback.")
        kws = fallback_frequency(combined['__corpus'], top_n=int(top_n), extra_stopwords=extra_stopwords)

    combined['Keywords'] = [k or "" for k in kws]

    non_empty = sum(bool(k.strip()) for k in combined['Keywords'])
    st.write(f"Records with at least one keyword: {non_empty} / {len(combined)}")

    st.subheader("Preview (first 100 rows)")
    st.dataframe(combined.drop(columns=['__corpus']).head(100))

    # Final download
    final_cols = ["Authors","Title","Year","Source","ArticleURL","Type","DOI","Abstract","FullTextURL","Keywords"]
    final_cols = [c for c in final_cols if c in combined.columns]
    final_df = combined[final_cols]
    csv_bytes = final_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download cleaned CSV", data=csv_bytes, file_name="collated_sources_cleaned.csv", mime="text/csv")

else:
    st.info("Please upload one or more .csv or .xlsx files to start.")
