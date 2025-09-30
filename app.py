import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
from datetime import datetime
import os
from openai import OpenAI
from collections import Counter
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.cluster import KMeans
import plotly.express as px

# Impostazioni del brand da analizzare
BRAND = "Boscolo Viaggi"
REVIEW_URLS = [
    "https://it.trustpilot.com/review/boscolo.com",
    "https://www.tripadvisor.it/Attraction_Review-g187867-d24108558-Reviews-Boscolo_Viaggi-Padua_Province_of_Padua_Veneto.html",
    "https://www.google.com/search?q=Boscolo+Tours+S.P.A.+Recensioni&sa=X"
]

# Attributi tipici per un tour operator
TOUR_OPERATOR_ASPECTS = [
    'itinerari', 'guida turistica', 'esperienza di viaggio', 'organizzazione del tour',
    'servizio clienti', 'trasporti', 'attività proposte'
]

# Configurazione delle librerie necessarie
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    st.error("❌ Plotly mancante: pip install plotly")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    import networkx as nx
    ML_CORE_AVAILABLE = True
except ImportError:
    ML_CORE_AVAILABLE = False
    st.error("❌ Scikit-learn mancanti: pip install scikit-learn networkx")

# Inizializzazione di SentenceTransformer per il Topic Modeling
sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
topic_model = BERTopic(embedding_model=sentence_model, nr_topics="auto", calculate_probabilities=True)

# Funzione per mostrare messaggi
def show_message(message, type="info", details=None):
    """Mostra messaggi stilizzati con dettagli opzionali"""
    if type == "success":
        st.markdown(f'<div class="success-box">✅ {message}</div>', unsafe_allow_html=True)
    elif type == "warning":
        st.markdown(f'<div class="warning-box">⚠️ {message}</div>', unsafe_allow_html=True)
    elif type == "error":
        st.markdown(f'<div class="error-box">❌ {message}</div>', unsafe_allow_html=True)
        if details:
            with st.expander("🔍 Dettagli Errore"):
                st.text(details)
    else:
        st.info(f"ℹ️ {message}")
    
    if details and type != "error":
        st.caption(f"💡 {details}")

# Funzione per analizzare le recensioni
def analyze_reviews(reviews_data):
    """Analizza le recensioni per il brand Boscolo Viaggi"""
    all_reviews = []

    # Combina recensioni da tutte le piattaforme
    for url in REVIEW_URLS:
        response = requests.get(url)
        # Simula l'analisi delle recensioni (qui dovresti usare un parser per estrarre le recensioni da ciascun link)
        reviews = extract_reviews_from_url(url)
        all_reviews.extend(reviews)
    
    if len(all_reviews) < 5:
        st.warning("⚠️ Servono almeno 5 recensioni per un'analisi efficace.")
    
    # Analizza i sentimenti multi-dimensionali (esempio semplificato)
    sentiment_results = analyze_sentiment(all_reviews)

    # Analisi dei topic
    topics, probabilities = topic_model.fit_transform([review['text'] for review in all_reviews])
    
    # Mostra risultati dell'analisi
    st.write("Analisi Sentiment:", sentiment_results)
    st.write("Analisi Topics:", topics)

# Funzione di analisi sentimenti (semplificata)
def analyze_sentiment(reviews):
    """Esegui una semplice analisi dei sentimenti sulle recensioni"""
    sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
    for review in reviews:
        if "bene" in review['text']:
            sentiments['positive'] += 1
        elif "male" in review['text']:
            sentiments['negative'] += 1
        else:
            sentiments['neutral'] += 1
    return sentiments

# Funzione per estrarre recensioni da un URL (simulata)
def extract_reviews_from_url(url):
    """Estrai recensioni da un URL specifico (simulato)"""
    # Questo è un esempio, dovrai fare scraping dai siti o usare API per ottenere recensioni reali.
    reviews = [
        {"text": "Ottimo tour, guida molto preparata!", "rating": 5},
        {"text": "Esperienza fantastica, tutto ben organizzato.", "rating": 4},
        {"text": "Servizio clienti un po' lento, ma la qualità del tour è buona.", "rating": 3},
    ]
    return reviews

# Main
def main():
    st.title(f"Analisi Recensioni {BRAND}")
    st.sidebar.markdown(f"Analisi delle recensioni di {BRAND}")

    if st.button("Avvia Analisi"):
        # Esegui analisi delle recensioni
        analyze_reviews(None)

if __name__ == "__main__":
    main()
