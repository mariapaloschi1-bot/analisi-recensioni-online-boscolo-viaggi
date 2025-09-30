import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
import pandas as pd
import time

# === CONFIG ===
BRAND = "Boscolo Viaggi"
TRUSTPILOT_URL = "https://it.trustpilot.com/review/boscolo.com"
TRIPADVISOR_URL = "https://www.tripadvisor.it/Attraction_Review-g187867-d24108558-Reviews-Boscolo_Viaggi-Padua_Province_of_Padua_Veneto.html"
MAX_REVIEWS = 10

# HuggingFace API key
HUGGINGFACE_API_KEY = st.secrets.get("HF_API_KEY") or os.getenv("HF_API_KEY", "")

# ---------- CSS STYLE ----------
st.markdown("""
    <style>
        html, body, .stApp {
            background-color: #f6f8fc;
        }
        .main-title {
            font-size:2.2em; font-weight:bold; color:#27334b; margin-bottom:0.2em;
        }
        .brand-logo {
            border-radius: 16px;
            margin-bottom: 10px;
        }
        .review-card {
            border-radius: 12px;
            background: #fff;
            box-shadow: 0 2px 12px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            padding: 1em 1.4em;
        }
        .sentiment-label {
            font-size: 1.1em;
            display:inline-block;
            margin-right:8px;
            font-weight: bold;
            padding: 2px 9px;
            border-radius: 8px;
        }
        .POS {
            background-color: #e6faef;
            color: #098b4f;
        }
        .NEU {
            background-color: #f6f7fa;
            color: #727272;
        }
        .NEG {
            background-color: #ffeaea;
            color: #c00e44;
        }
        .review-portal {
            font-size: 0.95em;
            color: #666;
            font-style: italic;
        }
        .stDataFrame {
            background: #fff !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- HUGGINGFACE ----------
def huggingface_sentiment(text, model="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": text}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

# ---------- SCRAPING ----------
def scrape_trustpilot_reviews(url, max_reviews=MAX_REVIEWS):
    reviews = []
    page = 1
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Bot/1.0)"}
    while len(reviews) < max_reviews:
        paged_url = url if page == 1 else f"{url}?page={page}"
        resp = requests.get(paged_url, headers=headers)
        if resp.status_code != 200:
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        blocks = soup.find_all("section", {"data-testid": "review-card"})
        if not blocks:
            break
        for block in blocks:
            p = block.find("p", {"data-testid": "review-content"})
            if p:
                reviews.append(p.get_text(strip=True))
            if len(reviews) >= max_reviews:
                break
        page += 1
        time.sleep(0.8)
    return reviews

def scrape_tripadvisor_reviews(url, max_reviews=MAX_REVIEWS):
    reviews = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Bot/1.0)"}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        return reviews
    soup = BeautifulSoup(resp.text, "html.parser")
    blocks = soup.find_all("q", {"class": "QewHA H4 _a"})
    for block in blocks:
        reviews.append(block.get_text(strip=True))
        if len(reviews) >= max_reviews:
            break
    return reviews

def analyze_sentiment_label(sentiment_json):
    # Support for HuggingFace output format
    # Can be [{'label':label, 'score':score}, ...] OR [[{'label':...}]]
    try:
        if isinstance(sentiment_json[0], list):
            label = sentiment_json[0][0]['label']
            score = sentiment_json[0][0]['score']
        else:
            label = sentiment_json[0]['label']
            score = sentiment_json[0]['score']
        if label.lower().startswith("pos"):
            label_cls, color = "POS", "POS"
        elif label.lower().startswith("neg"):
            label_cls, color = "NEG", "NEG"
        else:
            label_cls, color = "NEU", "NEU"
        return label_cls, score
    except Exception:
        return "NEU", 0.0

# ---------- STREAMLIT UI ----------
st.markdown(f"<div class='main-title'>🧳 Analisi Recensioni Online – {BRAND}</div>", unsafe_allow_html=True)
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"🔗 [Trustpilot]({TRUSTPILOT_URL}) &nbsp; | &nbsp; [TripAdvisor]({TRIPADVISOR_URL})")
    st.caption("Premi il bottone per raccogliere e analizzare le ultime recensioni reali!")
with col2:
    st.image("https://www.boscolo.com/wp-content/uploads/2023/03/boscolo-logo.svg", width=160, caption="", output_format="PNG", use_column_width=False, channels="RGB", clamp=True)

st.write("")

if st.button("🚀 Avvia Analisi Automatica", type="primary"):
    with st.spinner("Scraping e analisi in corso..."):
        trustpilot_reviews = scrape_trustpilot_reviews(TRUSTPILOT_URL, MAX_REVIEWS)
        tripadvisor_reviews = scrape_tripadvisor_reviews(TRIPADVISOR_URL, MAX_REVIEWS)

        all_results = []
        st.markdown("### 🌟 Recensioni Trustpilot")
        for idx, rec in enumerate(trustpilot_reviews):
            try:
                sentiment = huggingface_sentiment(rec)
                label, score = analyze_sentiment_label(sentiment)
            except Exception as e:
                label, score = "NEU", 0.0
            with st.container():
                st.markdown(f"""
                <div class='review-card'>
                    <span class='sentiment-label {label}'>{label}</span>
                    <span class='review-portal'>Trustpilot</span>
                    <br>
                    <span style='font-size:1.03em'>{rec}</span>
                </div>""", unsafe_allow_html=True)
            all_results.append({
                "Piattaforma": "Trustpilot",
                "Recensione": rec,
                "Sentiment": label,
                "Confidenza": round(score, 3)
            })

        st.markdown("### ✈️ Recensioni TripAdvisor")
        for idx, rec in enumerate(tripadvisor_reviews):
            try:
                sentiment = huggingface_sentiment(rec)
                label, score = analyze_sentiment_label(sentiment)
            except Exception as e:
                label, score = "NEU", 0.0
            with st.container():
                st.markdown(f"""
                <div class='review-card'>
                    <span class='sentiment-label {label}'>{label}</span>
                    <span class='review-portal'>TripAdvisor</span>
                    <br>
                    <span style='font-size:1.03em'>{rec}</span>
                </div>""", unsafe_allow_html=True)
            all_results.append({
                "Piattaforma": "TripAdvisor",
                "Recensione": rec,
                "Sentiment": label,
                "Confidenza": round(score, 3)
            })

        # Tabella riepilogativa
        st.markdown("## 📊 Tabella riepilogativa")
        df = pd.DataFrame(all_results)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("💾 Scarica risultati in CSV", df.to_csv(index=False), "analisi_boscolo.csv", "text/csv")

        # Statistiche rapide
        st.markdown("### 📈 Statistiche veloci")
        col1, col2, col3 = st.columns(3)
        tot = len(df)
        pos = (df['Sentiment'] == 'POS').sum()
        neg = (df['Sentiment'] == 'NEG').sum()
        neu = (df['Sentiment'] == 'NEU').sum()
        with col1: st.metric("Totale", tot)
        with col2: st.metric("Positive", pos)
        with col3: st.metric("Negative", neg)
        st.progress(pos/tot if tot else 0.01)

else:
    st.info("Premi il bottone qui sopra per avviare lo scraping e l'analisi delle recensioni.")

st.caption("Grafica migliorata • Powered by HuggingFace, Streamlit, BeautifulSoup © 2024")
