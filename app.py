import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
import time

# === CONFIG ===
BRAND = "Boscolo Viaggi"
TRUSTPILOT_URL = "https://it.trustpilot.com/review/boscolo.com"
TRIPADVISOR_URL = "https://www.tripadvisor.it/Attraction_Review-g187867-d24108558-Reviews-Boscolo_Viaggi-Padua_Province_of_Padua_Veneto.html"
MAX_REVIEWS = 10

# Leggi la HuggingFace API key dai secrets (Streamlit Cloud: st.secrets["HF_API_KEY"])
HUGGINGFACE_API_KEY = st.secrets.get("HF_API_KEY") or os.getenv("HF_API_KEY", "")

# === HUGGINGFACE API ===
def huggingface_sentiment(text, model="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": text}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

# === SCRAPING TRUSTPILOT ===
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
        time.sleep(1)
    return reviews

# === SCRAPING TRIPADVISOR ===
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

# === STREAMLIT UI ===
st.set_page_config(page_title=f"Recensioni Online {BRAND}", page_icon="🧳", layout="wide")
st.title(f"🧳 Analisi Recensioni Online – {BRAND}")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"""
    - [Trustpilot]({TRUSTPILOT_URL})
    - [TripAdvisor]({TRIPADVISOR_URL})
    """)
with col2:
    st.image("https://www.boscolo.com/wp-content/uploads/2023/03/boscolo-logo.svg", width=180, caption="Boscolo Viaggi")

st.info("Clicca il bottone per raccogliere e analizzare automaticamente le ultime recensioni dai portali ufficiali!")

if st.button("🚀 Avvia Analisi Automatica"):
    with st.spinner("Scraping recensioni Trustpilot..."):
        trustpilot_reviews = scrape_trustpilot_reviews(TRUSTPILOT_URL, MAX_REVIEWS)
    with st.spinner("Scraping recensioni TripAdvisor..."):
        tripadvisor_reviews = scrape_tripadvisor_reviews(TRIPADVISOR_URL, MAX_REVIEWS)

    all_results = []
    st.markdown("## Risultati Trustpilot")
    for idx, rec in enumerate(trustpilot_reviews):
        try:
            sentiment = huggingface_sentiment(rec)
            label = sentiment[0][0]['label'] if isinstance(sentiment[0], list) else sentiment[0]['label']
            score = sentiment[0][0]['score'] if isinstance(sentiment[0], list) else sentiment[0]['score']
        except Exception as e:
            label, score = "Errore", "-"
        st.write(f"**{idx+1}.** {rec}")
        st.write(f":mag: Sentiment: **{label}** | Score: {score}")
        all_results.append({
            "portal": "Trustpilot",
            "review": rec,
            "sentiment": label,
            "confidence": score
        })
        st.markdown("---")

    st.markdown("## Risultati TripAdvisor")
    for idx, rec in enumerate(tripadvisor_reviews):
        try:
            sentiment = huggingface_sentiment(rec)
            label = sentiment[0][0]['label'] if isinstance(sentiment[0], list) else sentiment[0]['label']
            score = sentiment[0][0]['score'] if isinstance(sentiment[0], list) else sentiment[0]['score']
        except Exception as e:
            label, score = "Errore", "-"
        st.write(f"**{idx+1}.** {rec}")
        st.write(f":mag: Sentiment: **{label}** | Score: {score}")
        all_results.append({
            "portal": "TripAdvisor",
            "review": rec,
            "sentiment": label,
            "confidence": score
        })
        st.markdown("---")

    # Tabella riassuntiva
    import pandas as pd
    if all_results:
        df = pd.DataFrame(all_results)
        st.markdown("## Tabella riepilogativa")
        st.dataframe(df, use_container_width=True)
        st.download_button("Scarica risultati CSV", df.to_csv(index=False), "analisi_boscolo.csv", "text/csv")

else:
    st.warning("Premi il bottone qui sopra per avviare lo scraping e l'analisi delle recensioni.")

st.caption("Powered by HuggingFace, Streamlit & BeautifulSoup | Codice pronto per GitHub & Streamlit Cloud")
