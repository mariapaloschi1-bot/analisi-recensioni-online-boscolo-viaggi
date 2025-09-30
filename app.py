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
import time
import random # Importato per generare dati casuali
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Configurazione pagina (DEVE essere la prima chiamata a st)
st.set_page_config(
    page_title="Boscolo Viaggi Reviews",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato
st.markdown("""
<style>
    /* Stili CSS (omessi per brevità, sono gli stessi di prima) */
    .stApp { background-color: #000000; color: #FFFFFF; }
    .main-header { text-align: center; padding: 20px; background: linear-gradient(135deg, #005691 0%, #0099FF 25%, #FFD700 75%, #8B5CF6 100%); border-radius: 20px; margin-bottom: 30px; }
    .stButton > button { background-color: #0099FF; color: #FFFFFF; border: none; }
    section[data-testid="stSidebar"] { background-color: #1A1A1A; }
</style>
""", unsafe_allow_html=True)

# --- STATO DELL'APPLICAZIONE ---
if 'reviews_data' not in st.session_state:
    st.session_state.reviews_data = {
        'trustpilot_reviews': [],
        'extended_reviews': {'total_count': 0, 'all_reviews': []},
        'analysis_results': None
    }
if 'data_imported' not in st.session_state:
    st.session_state.data_imported = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# --- FUNZIONI HELPER ---
def show_message(message, type="info"):
    if type == "success":
        st.success(f"✅ {message}")
    elif type == "warning":
        st.warning(f"⚠️ {message}")
    else:
        st.info(f"ℹ️ {message}")

# --- FUNZIONI MOCK (Simulate) ---
def fetch_trustpilot_reviews(url, limit):
    time.sleep(1.5)
    # NUOVO: Genera un numero di recensioni pari a 'limit'
    reviews = []
    for i in range(limit):
        reviews.append({
            'rating': random.randint(1, 5),
            'review_text': f'Questa è una recensione simulata numero {i+1}.',
            'user': {'name': f'Utente Mock {i+1}'},
            'timestamp': '2025-01-01'
        })
    return reviews

def fetch_google_extended_reviews(name, limit):
    time.sleep(1.5)
    # NUOVO: Genera un numero di recensioni pari a 'limit'
    reviews = []
    for i in range(limit):
        reviews.append({
            'rating': random.randint(2, 5),
            'review_text': f'Recensione estesa simulata numero {i+1}.',
            'review_source': 'Mock Source'
        })
    return {'total_count': limit, 'all_reviews': reviews}

def analyze_reviews(reviews, source):
    # Simula un'analisi un po' più realistica
    if not reviews:
        return {'total': 0, 'avg_rating': 0, 'sample_strengths': []}
    
    total_rating = sum(r['rating'] for r in reviews)
    avg_rating = round(total_rating / len(reviews), 2)
    
    return {'total': len(reviews), 'avg_rating': avg_rating, 'sample_strengths': ['Guida esperta', 'Itinerario ben strutturato', 'Buon rapporto qualità/prezzo']}

# --- INTERFACCIA PRINCIPALE ---
st.markdown("<h1 class='main-header'>✈️ REVIEWS: Boscolo Viaggi by Maria</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.info("Utilizza questa dashboard per importare e analizzare le recensioni.")

tab1, tab2, tab3 = st.tabs(["🌍 Multi-Platform Import", "📊 Cross-Platform Analysis", "🤖 AI Strategic Insights"])

with tab1:
    st.markdown("### 🌍 Data Import - Target: Boscolo Viaggi")

    if st.session_state.data_imported:
        total_tp = len(st.session_state.reviews_data['trustpilot_reviews'])
        total_ext = st.session_state.reviews_data['extended_reviews']['total_count']
        show_message(f"Dati importati: {total_tp} da Trustpilot, {total_ext} da Extended Reviews. Ora puoi avviare l'analisi.", "success")
    
    if st.session_state.analysis_done:
        show_message("Analisi completata! Vai al tab 'Cross-Platform Analysis' per vedere i risultati.", "success")

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🌟 Trustpilot", expanded=True):
            trustpilot_url = st.text_input("URL Trustpilot", value="https://it.trustpilot.com/review/boscolo.com")
            # Lo slider ora controlla il numero di recensioni generate
            tp_limit = st.slider("Max recensioni Trustpilot", 50, 2000, 1500, key="tp_limit")
            if st.button("📥 Import Trustpilot", use_container_width=True):
                with st.spinner(f"Importo {tp_limit} recensioni da Trustpilot..."):
                    st.session_state.reviews_data['trustpilot_reviews'] = fetch_trustpilot_reviews(trustpilot_url, tp_limit)
                    st.session_state.data_imported = True
                    st.session_state.analysis_done = False
                st.rerun()

    with col2:
        with st.expander("🔍 Extended Reviews", expanded=True):
            business_name = st.text_input("Nome Business", value="Boscolo Viaggi")
            # NUOVO: Slider anche per le recensioni estese
            ext_limit = st.slider("Max recensioni Extended", 50, 2000, 1500, key="ext_limit")
            if st.button("📥 Import Extended Reviews", use_container_width=True):
                with st.spinner(f"Importo {ext_limit} recensioni estese..."):
                    st.session_state.reviews_data['extended_reviews'] = fetch_google_extended_reviews(business_name, ext_limit)
                    st.session_state.data_imported = True
                    st.session_state.analysis_done = False
                st.rerun()

    st.divider()

    if st.session_state.data_imported:
        if st.button("📊 Avvia Analisi Multi-Platform", type="primary", use_container_width=True):
            with st.spinner("Analisi in corso..."):
                analysis_results = {}
                if st.session_state.reviews_data['trustpilot_reviews']:
                    analysis_results['trustpilot'] = analyze_reviews(st.session_state.reviews_data['trustpilot_reviews'], 'trustpilot')
                if st.session_state.reviews_data['extended_reviews']['all_reviews']:
                    analysis_results['extended'] = analyze_reviews(st.session_state.reviews_data['extended_reviews']['all_reviews'], 'extended')
                
                st.session_state.reviews_data['analysis_results'] = analysis_results
                st.session_state.analysis_done = True
            st.rerun()
    else:
        st.warning("Importa i dati da almeno una piattaforma per poter avviare l'analisi.")

with tab2:
    st.header("📊 Risultati Analisi Cross-Platform")
    if st.session_state.analysis_done and st.session_state.reviews_data['analysis_results']:
        results = st.session_state.reviews_data['analysis_results']
        
        st.subheader("Riepilogo Totale")
        total_reviews_analyzed = sum(res['total'] for res in results.values())
        avg_rating_overall = sum(res['avg_rating'] * res['total'] for res in results.values()) / total_reviews_analyzed if total_reviews_analyzed > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Totale Recensioni Analizzate", f"{total_reviews_analyzed}")
        with col2:
            st.metric("Rating Medio Complessivo", f"{avg_rating_overall:.2f} ⭐")

        st.divider()

        if 'trustpilot' in results:
            with st.container(border=True):
                st.subheader("🌟 Trustpilot Insights")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Recensioni Analizzate (TP)", results['trustpilot']['total'])
                with col2:
                    st.metric("Rating Medio (TP)", f"{results['trustpilot']['avg_rating']} ⭐")
                st.write("**Punti di Forza (Esempio):**")
                for strength in results['trustpilot']['sample_strengths']:
                    st.success(f"• {strength}")
        
        if 'extended' in results:
            with st.container(border=True):
                st.subheader("🔍 Extended Reviews Insights")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Recensioni Analizzate (Ext)", results['extended']['total'])
                with col2:
                    st.metric("Rating Medio (Ext)", f"{results['extended']['avg_rating']} ⭐")
                st.write("**Punti di Forza (Esempio):**")
                for strength in results['extended']['sample_strengths']:
                    st.success(f"• {strength}")
    else:
        st.info("Nessuna analisi ancora eseguita. Vai al tab 'Multi-Platform Import', carica i dati e avvia l'analisi.")

with tab3:
    st.header("🤖 AI Strategic Insights")
    st.info("Questa sezione mostrerà gli insight strategici generati dall'AI una volta completata l'analisi.")
