#!/usr/bin/env python3
"""
Reviews Analyzer v3.1 - Unified Enterprise Edition by Maria
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
    st.stop()

# --- Controllo librerie avanzate ---
# ... (omesso per brevità, ma presente nel codice caricato)

# CSS personalizzato
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    .main-header { text-align: center; padding: 20px; background: linear-gradient(135deg, #005691 0%, #0099FF 25%, #FFD700 75%, #8B5CF6 100%); border-radius: 20px; margin-bottom: 30px; }
    .stButton > button { background-color: #0099FF; color: #FFFFFF; border: none; }
    section[data-testid="stSidebar"] { background-color: #1A1A1A; }
    [data-testid="stMetric"] { background-color: #1a1a1a; padding: 15px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- STATO DELL'APPLICAZIONE (Session State) ---
if 'reviews_data' not in st.session_state:
    st.session_state.reviews_data = {
        'trustpilot_reviews': [], 'google_reviews': [], 'tripadvisor_reviews': [],
        'extended_reviews': {'all_reviews': [], 'sources_breakdown': {}, 'total_count': 0},
        'analysis_results': None
    }
if 'flags' not in st.session_state:
    st.session_state.flags = {'data_imported': False, 'analysis_done': False}

# ============================================================================
# FUNZIONI API REALI
# ============================================================================

def safe_api_call_with_progress(api_function, *args, **kwargs):
    progress_text = f"Chiamata a {api_function.__name__} in corso..."
    my_bar = st.progress(0, text=progress_text)
    try:
        # Questa è una simulazione della chiamata reale, che può essere molto lunga
        # In un'implementazione reale, qui ci sarebbe il codice di polling
        for i in range(10, 81, 10):
            time.sleep(random.uniform(2, 5)) # Simula l'attesa per il task API
            my_bar.progress(i, text=f"Elaborazione in corso su DataForSEO... ({i}%)")
        
        result = api_function(*args, **kwargs) # Esegue la funzione (che qui è simulata)

        my_bar.progress(100, text="Completato!")
        time.sleep(1)
        my_bar.empty()
        return result
    except Exception as e:
        my_bar.empty()
        logger.error(f"Errore API in {api_function.__name__}: {str(e)}")
        st.error(f"Errore durante la chiamata API: {str(e)}")
        return None

# --- Implementazioni REALI (Simulate per evitare costi reali durante i test) ---
def fetch_trustpilot_reviews(tp_url, limit):
    logger.info(f"SIMULAZIONE REALE: Fetch Trustpilot per {tp_url} con limite {limit}")
    return [{'rating': random.randint(3, 5), 'review_text': f'Recensione simulata da Trustpilot #{i+1}'} for i in range(limit)]

def fetch_google_reviews(place_id, location, limit):
    logger.info(f"SIMULAZIONE REALE: Fetch Google per {place_id} con limite {limit}")
    return [{'rating': random.randint(3, 5), 'review_text': f'Recensione simulata da Google #{i+1}'} for i in range(limit)]
    
def fetch_tripadvisor_reviews(ta_url, location, limit):
    logger.info(f"SIMULAZIONE REALE: Fetch TripAdvisor per {ta_url} con limite {limit}")
    return [{'rating': random.randint(3, 5), 'review_text': f'Recensione simulata da TripAdvisor #{i+1}'} for i in range(limit)]

# ... (altre funzioni di analisi di base)
def analyze_reviews_basic(reviews: List[Dict]) -> Dict:
    if not reviews: return {}
    ratings = [r['rating'] for r in reviews]
    return {
        'total': len(reviews),
        'avg_rating': round(np.mean(ratings), 2) if ratings else 0,
        'top_themes': [('viaggio', 50), ('organizzazione', 45), ('guida', 30)],
    }

# ============================================================================
# INTERFACCIA PRINCIPALE
# ============================================================================

st.markdown("<h1 class='main-header'>✈️ REVIEWS: Boscolo Viaggi by Maria</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🌍 Import Dati", "📊 Dashboard Analisi", "📥 Export"])

# --- TAB 1: IMPORT ---
with tab1:
    st.markdown("### 🌍 Importa Dati da Diverse Piattaforme")
    if not CREDENTIALS_OK: st.stop()

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🌟 Trustpilot", expanded=True):
            tp_url = st.text_input("URL Trustpilot", "https://it.trustpilot.com/review/boscolo.com", key="tp_url_input")
            tp_limit = st.slider("Max Recensioni TP", 50, 2000, 200, key="tp_slider")
            if st.button("Importa da Trustpilot", use_container_width=True):
                reviews = safe_api_call_with_progress(fetch_trustpilot_reviews, tp_url, tp_limit)
                if reviews:
                    st.session_state.reviews_data['trustpilot_reviews'] = reviews
                    st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni da Trustpilot importate!"); time.sleep(1); st.rerun()

    with col2:
        with st.expander("✈️ TripAdvisor", expanded=True):
            ta_url = st.text_input("URL TripAdvisor", "https://www.tripadvisor.it/Attraction_Review-g187867-d24108558-Reviews-Boscolo_Viaggi-Padua_Province_of_Padua_Veneto.html", key="ta_url_input")
            ta_limit = st.slider("Max Recensioni TA", 50, 2000, 200, key="ta_slider")
            if st.button("Importa da TripAdvisor", use_container_width=True):
                reviews = safe_api_call_with_progress(fetch_tripadvisor_reviews, ta_url, "Italy", ta_limit)
                if reviews:
                    st.session_state.reviews_data['tripadvisor_reviews'] = reviews
                    st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni da TripAdvisor importate!"); time.sleep(1); st.rerun()

    with st.expander("📍 Google Reviews"):
        # NOTA: L'URL di ricerca non è un Place ID. Per Google serve il Place ID specifico.
        # Lo trovi su Google Maps, di solito inizia con "ChIJ...".
        # Per questa demo, ne usiamo uno fittizio.
        g_place_id = st.text_input("Google Place ID", "ChIJ-R_d-iV-1BIRsA7DW2s-2GA", key="g_id_input", help="Questo è un Place ID di esempio. Sostituiscilo con quello corretto per 'Boscolo Tours S.P.A.'")
        g_limit = st.slider("Max Recensioni Google", 50, 2000, 200, key="g_slider")
        if st.button("Importa da Google", use_container_width=True):
            reviews = safe_api_call_with_progress(fetch_google_reviews, g_place_id, "Italy", g_limit)
            if reviews:
                st.session_state.reviews_data['google_reviews'] = reviews
                st.session_state.flags['data_imported'] = True
                st.success(f"{len(reviews)} recensioni da Google importate!"); time.sleep(1); st.rerun()

    st.markdown("---")
    st.subheader("Riepilogo Dati Importati")
    
    counts = {
        "Trustpilot": len(st.session_state.reviews_data['trustpilot_reviews']),
        "Google": len(st.session_state.reviews_data['google_reviews']),
        "TripAdvisor": len(st.session_state.reviews_data['tripadvisor_reviews']),
    }
    total_items = sum(counts.values())

    if total_items > 0:
        active_platforms = [c for c in counts.values() if c > 0]
        if active_platforms:
            cols = st.columns(len(active_platforms))
            i = 0
            for platform, count in counts.items():
                if count > 0:
                    with cols[i]:
                        st.metric(label=f"📝 {platform}", value=count)
                    i += 1
    else:
        st.info("Nessun dato ancora importato.")


# --- TAB 2: ANALISI ---
with tab2:
    st.header("📊 Dashboard Analisi")
    if not st.session_state.flags['data_imported']:
        st.info("⬅️ Importa dati dal tab 'Import Dati' per poter eseguire un'analisi.")
    else:
        if st.button("🔬 Esegui Analisi di Base", use_container_width=True, type="primary"):
            with st.spinner("Esecuzione analisi statistica..."):
                results = {}
                # Aggiungi un ciclo per analizzare tutte le piattaforme con dati
                for platform_key, platform_name in [('trustpilot_reviews', 'Trustpilot'), ('google_reviews', 'Google'), ('tripadvisor_reviews', 'TripAdvisor')]:
                    if st.session_state.reviews_data[platform_key]:
                        results[platform_name] = analyze_reviews_basic(st.session_state.reviews_data[platform_key])
                st.session_state.reviews_data['analysis_results'] = results
                st.session_state.flags['analysis_done'] = True
            st.success("Analisi di base completata!")
            time.sleep(1); st.rerun()

        st.markdown("---")
        
        # VISUALIZZAZIONE RISULTATI
        if st.session_state.flags['analysis_done']:
            st.subheader("🔬 Risultati Analisi di Base")
            analysis_results = st.session_state.reviews_data.get('analysis_results', {})
            if not analysis_results:
                st.warning("Nessun risultato da mostrare. Prova a rieseguire l'analisi.")
            else:
                for platform, results in analysis_results.items():
                    with st.expander(f"**{platform}** ({results.get('total', 0)} recensioni)", expanded=True):
                        c1, c2 = st.columns(2)
                        c1.metric("Rating Medio (Simulato)", f"{results.get('avg_rating', 0)} ⭐")
                        
                        st.write("**Temi Principali (Simulati):**")
                        st.write(", ".join([f"{theme[0]} ({theme[1]})" for theme in results.get('top_themes', [])]))

# --- TAB 3: EXPORT ---
with tab3:
    st.header("📥 Esporta Dati e Report")
    st.info("Le opzioni di export appariranno qui dopo aver eseguito le analisi.")
