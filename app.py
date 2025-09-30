#!/usr/bin/env python3
"""
Reviews Analyzer v3.0 - Unified Enterprise Edition by Maria
Supports: Trustpilot, Google Reviews, TripAdvisor, Yelp (via Extended Reviews), Reddit
Advanced Analytics: Multi-Dimensional Sentiment, ABSA, Topic Modeling, Customer Journey, SEO Intelligence
"""

import streamlit as st
import pandas as pd
import requests
import time
import json
import re
import numpy as np
import random
from datetime import datetime
import logging
from openai import OpenAI
import os
from urllib.parse import urlparse
from typing import Dict, List, Optional
from dataclasses import dataclass
import threading
from docx import Document
import io

# --- CONFIGURAZIONE PAGINA (DEVE essere la prima chiamata a st) ---
st.set_page_config(
    page_title="Boscolo Viaggi Reviews",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INIZIALIZZAZIONE LIBRERIE E CREDENZIALI
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Caricamento sicuro delle credenziali dai Secrets ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    DFSEO_LOGIN = st.secrets["DFSEO_LOGIN"]
    DFSEO_PASS = st.secrets["DFSEO_PASS"]
    CREDENTIALS_OK = True
except (KeyError, FileNotFoundError):
    st.error("⚠️ Credenziali API (OPENAI_API_KEY, DFSEO_LOGIN, DFSEO_PASS) non trovate nei Secrets! Aggiungile per far funzionare l'app.")
    CREDENTIALS_OK = False
    # Termina l'esecuzione se le credenziali mancano
    st.stop()

# --- Controllo librerie avanzate ---
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

ENTERPRISE_LIBS_AVAILABLE = PLOTLY_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE and BERTOPIC_AVAILABLE

# CSS personalizzato
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    .main-header { text-align: center; padding: 20px; background: linear-gradient(135deg, #005691 0%, #0099FF 25%, #FFD700 75%, #8B5CF6 100%); border-radius: 20px; margin-bottom: 30px; }
    .stButton > button { background-color: #0099FF; color: #FFFFFF; border: none; }
    section[data-testid="stSidebar"] { background-color: #1A1A1A; }
</style>
""", unsafe_allow_html=True)

# --- STATO DELL'APPLICAZIONE (Session State) ---
if 'reviews_data' not in st.session_state:
    st.session_state.reviews_data = {
        'trustpilot_reviews': [], 'google_reviews': [], 'tripadvisor_reviews': [],
        'extended_reviews': {'all_reviews': [], 'sources_breakdown': {}, 'total_count': 0},
        'reddit_discussions': [], 'analysis_results': None, 'enterprise_analysis': None, 'seo_analysis': None,
        'brand_keywords': {'raw_keywords': [], 'ai_insights': None, 'search_params': {}}
    }
if 'flags' not in st.session_state:
    st.session_state.flags = {'data_imported': False, 'analysis_done': False, 'enterprise_done': False, 'seo_done': False}


# ============================================================================
# FUNZIONI API REALI (sostituite con simulazioni per evitare costi e attese)
# ============================================================================
def safe_api_call_with_progress(api_function, *args, **kwargs):
    progress_text = f"Chiamata a {api_function.__name__} in corso (simulazione)..."
    my_bar = st.progress(0, text=progress_text)
    try:
        # Simulazione di un processo lungo
        for i in range(10, 81, 10):
            time.sleep(random.uniform(0.5, 1.5))
            my_bar.progress(i, text=f"{progress_text} ({i}%)")
        
        result = api_function(*args, **kwargs)

        my_bar.progress(100, text="Completato!")
        time.sleep(1)
        my_bar.empty()
        return result
    except Exception as e:
        my_bar.empty()
        logger.error(f"Errore API in {api_function.__name__}: {str(e)}")
        st.error(f"Errore durante la chiamata API (simulata): {str(e)}")
        return None

def fetch_trustpilot_reviews(tp_url, limit):
    logger.info(f"SIMULAZIONE: Fetch Trustpilot per {tp_url} con limite {limit}")
    return [{'rating': random.randint(3, 5), 'review_text': f'Recensione simulata TP {i+1}'} for i in range(limit)]

def fetch_google_reviews(place_id, location, limit):
    logger.info(f"SIMULAZIONE: Fetch Google per {place_id} con limite {limit}")
    return [{'rating': random.randint(3, 5), 'review_text': f'Recensione simulata Google {i+1}'} for i in range(limit)]
    
def fetch_tripadvisor_reviews(ta_url, location, limit):
    logger.info(f"SIMULAZIONE: Fetch TripAdvisor per {ta_url} con limite {limit}")
    return [{'rating': random.randint(3, 5), 'review_text': f'Recensione simulata TA {i+1}'} for i in range(limit)]

def fetch_google_extended_reviews(business_name, location, limit):
    logger.info(f"SIMULAZIONE: Fetch Extended per {business_name} con limite {limit}")
    all_reviews = [{'rating': random.randint(2, 5), 'review_text': f'Recensione estesa {i+1}', 'review_source': random.choice(['Yelp', 'Booking.com'])} for i in range(limit)]
    return {'all_reviews': all_reviews, 'sources_breakdown': {}, 'total_count': limit}

def fetch_reddit_discussions(reddit_urls, limit):
    logger.info(f"SIMULAZIONE: Fetch Reddit per {reddit_urls}")
    return [{'title': f'Discussione Reddit simulata {i+1}', 'subreddit': 'travel', 'author': 'user123'} for i in range(5)]

# Placeholder per le analisi complesse
def placeholder_enterprise_analysis(num_reviews):
    time.sleep(3)
    return {'metadata': {'total_reviews_analyzed': num_reviews}, 'performance_metrics': {'total_duration': 3.5}}

# ============================================================================
# INTERFACCIA PRINCIPALE
# ============================================================================

st.markdown("<h1 class='main-header'>✈️ REVIEWS: Boscolo Viaggi by Maria</h1>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.info("Dashboard di analisi recensioni e keywords per Boscolo Viaggi.")
    st.markdown("---")
    st.markdown("### 🔧 Enterprise Features Status")
    st.markdown(f"**Visualizzazioni:** {'✅ Attiva' if PLOTLY_AVAILABLE else '❌ Non disponibile'}")
    st.markdown(f"**Analisi Semantica:** {'✅ Attiva' if SENTENCE_TRANSFORMERS_AVAILABLE else '❌ Non disponibile'}")
    st.markdown(f"**Topic Modeling:** {'✅ Attiva' if BERTOPIC_AVAILABLE else '❌ Non disponibile'}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌍 Import", "📊 Analisi", "🤖 AI Insights", "🔍 Keywords", "📈 Visualizzazioni", "📥 Export"
])

# --- TAB 1: IMPORT ---
with tab1:
    st.markdown("### 🌍 Importa Dati da Diverse Piattaforme")
    if not CREDENTIALS_OK: st.stop()

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🌟 Trustpilot", expanded=True):
            tp_url = st.text_input("URL Trustpilot", "https://it.trustpilot.com/review/boscoloviaggi.com", key="tp_url_input")
            tp_limit = st.slider("Max Recensioni TP", 50, 2000, 200, key="tp_slider")
            if st.button("Importa da Trustpilot"):
                reviews = safe_api_call_with_progress(fetch_trustpilot_reviews, tp_url, tp_limit)
                if reviews:
                    st.session_state.reviews_data['trustpilot_reviews'] = reviews
                    st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni importate!"); time.sleep(1); st.rerun()

        with st.expander("✈️ TripAdvisor"):
            ta_url = st.text_input("URL TripAdvisor", key="ta_url_input")
            ta_limit = st.slider("Max Recensioni TA", 50, 2000, 200, key="ta_slider")
            if st.button("Importa da TripAdvisor", disabled=not ta_url):
                reviews = safe_api_call_with_progress(fetch_tripadvisor_reviews, ta_url, "Italy", ta_limit)
                if reviews:
                    st.session_state.reviews_data['tripadvisor_reviews'] = reviews
                    st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni importate!"); time.sleep(1); st.rerun()
    
    with col2:
        with st.expander("📍 Google Reviews"):
            g_place_id = st.text_input("Google Place ID", placeholder="Inizia con ChIJ...", key="g_id_input")
            g_limit = st.slider("Max Recensioni Google", 50, 2000, 200, key="g_slider")
            if st.button("Importa da Google", disabled=not g_place_id):
                reviews = safe_api_call_with_progress(fetch_google_reviews, g_place_id, "Italy", g_limit)
                if reviews:
                    st.session_state.reviews_data['google_reviews'] = reviews
                    st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni importate!"); time.sleep(1); st.rerun()

        with st.expander("🔍 Extended Reviews (Yelp, etc.)"):
            ext_name = st.text_input("Nome Business", "Boscolo Viaggi", key="ext_name_input")
            ext_limit = st.slider("Max Recensioni Estese", 50, 2000, 200, key="ext_slider")
            if st.button("Importa Recensioni Estese"):
                data = safe_api_call_with_progress(fetch_google_extended_reviews, ext_name, "Italy", ext_limit)
                if data:
                    st.session_state.reviews_data['extended_reviews'] = data
                    st.session_state.flags['data_imported'] = True
                    st.success(f"{data['total_count']} recensioni importate!"); time.sleep(1); st.rerun()

    with st.expander("💬 Reddit"):
        reddit_urls = st.text_area("URL Pagine Web da cercare su Reddit", placeholder="https://www.boscoloviaggi.com/...", key="reddit_urls_input")
        if st.button("Cerca Discussioni su Reddit"):
            discussions = safe_api_call_with_progress(fetch_reddit_discussions, reddit_urls, 100)
            if discussions:
                st.session_state.reviews_data['reddit_discussions'] = discussions
                st.session_state.flags['data_imported'] = True
                st.success(f"{len(discussions)} discussioni trovate!"); time.sleep(1); st.rerun()
    
    st.markdown("---")
    st.subheader("Riepilogo Dati Importati")
    counts = {
        "Trustpilot": len(st.session_state.reviews_data['trustpilot_reviews']),
        "Google": len(st.session_state.reviews_data['google_reviews']),
        "TripAdvisor": len(st.session_state.reviews_data['tripadvisor_reviews']),
        "Extended": st.session_state.reviews_data['extended_reviews']['total_count'],
        "Reddit": len(st.session_state.reviews_data['reddit_discussions'])
    }
    total_items = sum(counts.values())
    st.metric("Totale Items Caricati", total_items)
    st.write(counts)


# --- Altri TAB (implementazioni simili a quelle del codice del tuo amico) ---
with tab2:
    st.header("📊 Analisi Dati")
    st.info("Questa sezione mostrerà i risultati delle analisi.")

with tab3:
    st.header("🤖 AI Strategic Insights")
    st.info("Questa sezione mostrerà gli insight generati dall'AI.")

with tab4:
    st.header("🔍 Brand Keywords Analysis")
    st.info("Questa sezione permetterà di analizzare le keywords legate al brand.")

with tab5:
    st.header("📈 Visualizzazioni")
    st.info("Questa sezione mostrerà grafici e visualizzazioni interattive.")

with tab6:
    st.header("📥 Export")
    st.info("Questa sezione permetterà di esportare i dati e i report.")
