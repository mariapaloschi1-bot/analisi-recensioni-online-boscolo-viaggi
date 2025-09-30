#!/usr/bin/env python3
"""
Reviews Analyzer v2.1 ENTERPRISE EDITION
Supporto: Trustpilot, Google Reviews, TripAdvisor, Yelp (via Extended Reviews), Reddit
Advanced Analytics: Multi-Dimensional Sentiment, ABSA, Topic Modeling, Customer Journey
OBIETTIVO: Analisi completa per il brand 'Boscolo Viaggi'
Autore: Antonio De Luca
"""

import streamlit as st
import pandas as pd
import requests
import time
import json
import re
import numpy as np
from datetime import datetime, timedelta
import logging
from docx import Document
from openai import OpenAI
from collections import Counter
import os
from urllib.parse import urlparse, parse_qs
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ============================================================================
# ENTERPRISE LIBRARIES - INIZIALIZZAZIONE ROBUSTA
# ============================================================================

# Flags di disponibilità
ENTERPRISE_LIBS_AVAILABLE = False
HDBSCAN_AVAILABLE = False
BERTOPIC_AVAILABLE = False
PLOTLY_AVAILABLE = False

# Step 1: Verifica Plotly (essenziale per visualizzazioni)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    go = None

# Step 2: Verifica librerie ML core
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    import networkx as nx
    ML_CORE_AVAILABLE = True
except ImportError:
    ML_CORE_AVAILABLE = False

# Step 3: Verifica Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Step 4: Verifica HDBSCAN (opzionale)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# Step 5: Verifica BERTopic
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

# Determina disponibilità enterprise complessiva
ENTERPRISE_LIBS_AVAILABLE = (
    PLOTLY_AVAILABLE and
    ML_CORE_AVAILABLE and
    SENTENCE_TRANSFORMERS_AVAILABLE and
    BERTOPIC_AVAILABLE
)

@dataclass
class EnterpriseAnalysisResult:
    """Struttura unificata per risultati enterprise"""
    sentiment_analysis: Dict
    aspect_analysis: Dict
    topic_modeling: Dict
    customer_journey: Dict
    similarity_analysis: Dict
    performance_metrics: Dict


# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configurazione pagina
st.set_page_config(
    page_title="Boscolo Viaggi NLZYR",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Credenziali API (MANTENUTE come nel codice originale)
DFSEO_LOGIN = os.getenv('DFSEO_LOGIN', 'maria.paloschi@filoblu.com')
DFSEO_PASS = os.getenv('DFSEO_PASS', '0366ead7a9ec0d18')
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# CSS personalizzato - Design Moderno Nero/Viola/Multi-platform
st.markdown("""
<style>
    /* FORZA SFONDO NERO SU TUTTO */
    .stApp {
        background-color: #000000;
    }
    .main {
        background-color: #000000;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #000000;
    }
    [data-testid="stHeader"] {
        background-color: #000000;
    }
    /* FORZA TESTO BIANCO SU TUTTO */
    .stApp, .stApp * {
        color: #FFFFFF;
    }
    /* Header principale */
    .main-header {
        text-align: center;
        padding: 30px;
        /* Colori per Boscolo (Blu/Azzurro/Oro/Viola) */
        background: linear-gradient(135deg, #005691 0%, #0099FF 25%, #FFD700 75%, #8B5CF6 100%);
        border-radius: 20px;
        margin-bottom: 40px;
    }
    /* DATAFRAME NERO */
    [data-testid="stDataFrame"] {
        background-color: #000000;
    }
    [data-testid="stDataFrame"] iframe {
        background-color: #000000;
        filter: invert(1);
    }
    /* TABS NERE */
    .stTabs {
        background-color: #000000;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #000000;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1A1A;
        color: #FFFFFF;
    }
    .stTabs [aria-selected="true"] {
        background-color: #000000;
        border-bottom: 2px solid #0099FF; /* Colore tema Boscolo */
    }
    /* BOTTONI */
    .stButton > button {
        background-color: #0099FF; /* Colore tema Boscolo */
        color: #FFFFFF;
        border: none;
    }
    /* INPUT */
    .stTextInput > div > div > input {
        background-color: #1A1A1A;
        color: #FFFFFF;
        border: 1px solid #3A3A3A;
    }
    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #1A1A1A;
    }
    /* METRICHE */
    [data-testid="metric-container"] {
        background-color: #1A1A1A;
        border: 1px solid #3A3A3A;
        border-radius: 10px;
        padding: 15px;
    }
    /* Colori Platform Badge */
    .badge-trustpilot { background-color: #00B67A !important; }
    .badge-google { background-color: #4285F4 !important; }
    .badge-tripadvisor { background-color: #00AF87 !important; }
    .badge-yelp { background-color: #D32323 !important; }
    .badge-reddit { background-color: #FF4500 !important; }
    .platform-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 5px;
        color: white !important;
        font-size: 0.8em;
        margin: 2px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- STATO DELL'APPLICAZIONE ESTESO ---
if 'reviews_data' not in st.session_state:
    st.session_state.reviews_data = {
        'trustpilot_reviews': [],
        'google_reviews': [],
        'tripadvisor_reviews': [],
        'extended_reviews': {'all_reviews': [], 'sources_breakdown': {}, 'total_count': 0},
        'reddit_discussions': [],
        'analysis_results': {},
        'ai_insights': "",
        'brand_keywords': {
            'raw_keywords': [],
            'filtered_keywords': [],
            'analysis_results': {},
            'ai_insights': {},
            'search_params': {}
        }
    }

# --- FUNZIONI HELPER (Mock/Placeholder per il file principale) ---
def show_message(message, type="info", details=None):
    if type == "success":
        st.markdown(f'<div class="success-box" style="padding: 10px; background-color: #1a4d3f; border-radius: 5px;">✅ {message}</div>', unsafe_allow_html=True)
    elif type == "warning":
        st.markdown(f'<div class="warning-box" style="padding: 10px; background-color: #4d401a; border-radius: 5px;">⚠️ {message}</div>', unsafe_allow_html=True)
    elif type == "error":
        st.markdown(f'<div class="error-box" style="padding: 10px; background-color: #4d1a1a; border-radius: 5px;">❌ {message}</div>', unsafe_allow_html=True)
        if details:
            with st.expander("🔍 Dettagli Errore"):
                st.text(details)
    else:
        st.info(f"ℹ️ {message}")

    if details and type != "error":
        st.caption(f"💡 {details}")

def create_metric_card(title, value, delta=None):
    with st.container():
        st.metric(title, value, delta)

def create_platform_badge(platform_name):
    platform_colors = {
        'trustpilot': 'badge-trustpilot',
        'google': 'badge-google',
        'tripadvisor': 'badge-tripadvisor',
        'reddit': 'badge-reddit',
        'yelp': 'badge-yelp',
        'extended': 'badge-yelp'
    }
    color_class = platform_colors.get(platform_name.lower(), 'platform-badge')
    return f'<span class="platform-badge {color_class}">{platform_name.title()}</span>'

def safe_api_call_with_progress(api_function, *args, **kwargs):
    # Mocking the actual API calls for simplicity in the display code
    with st.spinner(f"Chiamata API a {api_function.__name__} in corso..."):
        time.sleep(1.5)

    if 'fetch_trustpilot_reviews' in api_function.__name__:
        return [{'rating': 5, 'review_text': 'Ottimo tour, la guida era fantastica!', 'user': {'name': 'Mock User 1'}, 'timestamp': '2025-01-01T00:00:00Z'}]
    elif 'fetch_google_extended_reviews' in api_function.__name__:
        return {'all_reviews': [{'rating': 4, 'review_text': 'Perfetto! L\'organizzazione era impeccabile.', 'review_source': 'Mock Yelp'}], 'sources_breakdown': {'Mock Yelp': 1}, 'total_count': 1}
    else:
        return []

# Placeholder Classes (Devono essere implementate completamente nel codice effettivo)
class DataForSEOKeywordsExtractor:
    def __init__(self, login: str, password: str):
        self.login = login
        self.password = password
    def get_keywords_for_keywords(self, seed_keywords: List[str], location_code: int = 2380, language_code: str = "it", include_terms: List[str] = None, exclude_terms: List[str] = None) -> Optional[pd.DataFrame]:
        return pd.DataFrame(columns=['keyword', 'search_volume', 'cpc', 'competition_level'])
    def get_search_volume(self, keywords: List[str], location_code: int = 2380, language_code: str = "it") -> Optional[pd.DataFrame]:
        return pd.DataFrame(columns=['keyword', 'search_volume', 'cpc', 'competition_level'])

class EnterpriseReviewsAnalyzer:
    def __init__(self, openai_client):
        self.client = openai_client
        self.business_aspects = {'tour_operator': ['itinerario', 'guida', 'destinazione', 'prezzo'], 'generale': ['servizio', 'qualità']}
    def run_enterprise_analysis(self, all_reviews_data: Dict) -> Dict:
        return {'error': 'Mock Analysis'}
    def _combine_all_reviews(self, reviews_data: Dict) -> List[Dict]:
        return []

# Placeholder API Fetch Functions (corpi minimi per non fallire)
def fetch_trustpilot_reviews(tp_url, limit=2000):
    return [{'rating': 5, 'review_text': 'Ottimo tour', 'user': {'name': 'Mock User'}, 'timestamp': '2025-01-01T00:00:00Z'}]
def fetch_google_reviews(place_id, location="Italy", limit=2000):
    return [{'rating': 4, 'review_text': 'Buon viaggio', 'user': {'name': 'Mock User'}, 'timestamp': '2025-01-01T00:00:00Z'}]
def fetch_tripadvisor_reviews(tripadvisor_url, location="Italy", limit=2000):
    return [{'rating': 3, 'review_text': 'Così così', 'user': {'name': 'Mock User'}, 'timestamp': '2025-01-01T00:00:00Z'}]
def fetch_google_extended_reviews(business_name, location="Italy", limit=2000):
    return {'all_reviews': [{'rating': 5, 'review_text': 'Perfetto!', 'review_source': 'Mock Yelp'}], 'sources_breakdown': {'Mock Yelp': 1}, 'total_count': 1}
def fetch_reddit_discussions(reddit_urls_input, subreddits=None, limit=1000):
    return [{'title': 'Discussione mock', 'text': 'Testo mock', 'subreddit': 'Mock Sub'}]
def verify_dataforseo_credentials():
    return True, {'money': {'balance': 999.99}}
def analyze_reviews(reviews, source):
    return {'total': len(reviews), 'avg_rating': 4.5, 'sentiment_distribution': {'positive': 1, 'neutral': 0, 'negative': 0}, 'sentiment_percentage': {'positive': 100.0, 'neutral': 0.0, 'negative': 0.0}, 'top_themes': [('guida', 5)], 'sample_strengths': ['Ottimo tour'], 'sample_pain_points': [], 'monthly_trends': {}, 'length_distribution': {}, 'rating_distribution': {}}
def analyze_reddit_discussions(reddit_data):
    return {'total': 1, 'sentiment_percentage': {'positive': 50.0, 'neutral': 50.0, 'negative': 0.0}, 'subreddit_breakdown': {'Mock Sub': 1}, 'top_topics': [], 'discussions_sample': []}
def analyze_reviews_for_seo(reviews, source):
    return {'total_reviews_analyzed': len(reviews)}
def analyze_seo_with_ai(seo_insights_data):
    return "Strategia SEO generica."
def analyze_brand_keywords_with_ai(keywords_data, brand_name):
    return "Analisi keyword generica."
def create_multiplatform_visualizations(reviews_data):
    return {}


# --- INTERFACCIA PRINCIPALE ---

# Header con nuovo design multi-platform
st.markdown("<h1 class='main-header'>✈️ REVIEWS NLZYR: Boscolo Viaggi Intelligence</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://www.boscoloviaggi.com/sites/all/themes/boscolo/images/svg/logo-boscolo-colori.svg", width=200)
    st.markdown("### 📊 Multi-Platform Dashboard")
    st.info("Utilizza questa dashboard per importare, analizzare e visualizzare le recensioni da diverse piattaforme.")

# Contenuto principale con tabs
tab1, tab2, tab3 = st.tabs(["🌍 Multi-Platform Import", "📊 Cross-Platform Analysis", "🤖 AI Strategic Insights"])

with tab1:
    st.markdown("### 🌍 Data Import - Target: Boscolo Viaggi")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔗 Platform URLs")
        with st.expander("🌟 Trustpilot", expanded=True):
            trustpilot_url = st.text_input("URL Trustpilot", value="https://it.trustpilot.com/review/boscoloviaggi.com", key="tp_url")
            tp_limit = st.slider("Max recensioni Trustpilot", 50, 2000, 200, key="tp_limit")
            if st.button("📥 Import Trustpilot", key="btn_tp_import", use_container_width=True):
                st.session_state.reviews_data['trustpilot_reviews'] = safe_api_call_with_progress(fetch_trustpilot_reviews, trustpilot_url, tp_limit)
                show_message(f"{len(st.session_state.reviews_data['trustpilot_reviews'])} recensioni Trustpilot importate (Mock)!", "success")
                st.rerun()

    with col2:
        st.markdown("#### 🆔 IDs & Names")
        with st.expander("🔍 Extended Reviews (Yelp + Multi)", expanded=True):
            business_name_ext = st.text_input("Nome Business", value="Boscolo Viaggi", key="ext_name")
            if st.button("📥 Import Extended Reviews", key="btn_ext_import", use_container_width=True):
                st.session_state.reviews_data['extended_reviews'] = safe_api_call_with_progress(fetch_google_extended_reviews, business_name_ext)
                show_message(f"{st.session_state.reviews_data['extended_reviews']['total_count']} Extended Reviews importate (Mock)!", "success")
                st.rerun()

    st.divider()

    total_reviews = len(st.session_state.reviews_data.get('trustpilot_reviews', [])) + st.session_state.reviews_data.get('extended_reviews', {}).get('total_count', 0)

    if total_reviews > 0:
        if st.button("📊 Avvia Analisi Multi-Platform", key="btn_multi_analysis", type="primary", use_container_width=True):
            with st.spinner("Analisi in corso..."):
                # Mock analysis
                st.session_state.reviews_data['analysis_results'] = {
                    'trustpilot_analysis': analyze_reviews(st.session_state.reviews_data['trustpilot_reviews'], 'trustpilot'),
                    'extended_analysis': analyze_reviews(st.session_state.reviews_data['extended_reviews']['all_reviews'], 'extended')
                }
                time.sleep(2)
            show_message("Analisi multi-platform completata (Mock)!", "success")
            st.rerun()
    else:
        st.warning("Importa dati da almeno una piattaforma per avviare l'analisi.")


with tab2:
    st.header("📊 Risultati Analisi Cross-Platform")
    if 'analysis_results' in st.session_state.reviews_data and st.session_state.reviews_data['analysis_results']:
        results = st.session_state.reviews_data['analysis_results']
        
        tp_analysis = results.get('trustpilot_analysis')
        if tp_analysis:
            st.subheader("🌟 Trustpilot Insights")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Recensioni Analizzate", tp_analysis['total'])
            with col2:
                st.metric("Rating Medio", f"{tp_analysis['avg_rating']} ⭐")
            st.write("**Punti di Forza (Esempio):**")
            for strength in tp_analysis['sample_strengths']:
                st.success(f"• {strength}")
    else:
        st.info("Nessuna analisi ancora eseguita. Vai al tab 'Multi-Platform Import', carica i dati e avvia l'analisi.")

with tab3:
    st.header("🤖 AI Strategic Insights")
    st.info("Questa sezione mostrerà gli insight strategici generati dall'AI una volta completata l'analisi.")


if __name__ == "__main__":
    logger.info("Reviews Analyzer Tool v2.1 avviato")
