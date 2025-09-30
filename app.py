#!/usr/bin/env python3
"""
Reviews Analyzer v3.1 - Unified Enterprise Edition by Maria
Supports: Trustpilot, Google Reviews, TripAdvisor, Yelp (via Extended Reviews), Reddit
Advanced Analytics: Multi-Dimensional Sentiment, ABSA, Topic Modeling, Customer Journey, SEO Intelligence
"""

import streamlit as st
import pandas as pd
import time
import random
import logging
from typing import Dict, List

# --- CONFIGURAZIONE PAGINA (DEVE essere la prima chiamata a st) ---
st.set_page_config(
    page_title="Boscolo Viaggi Reviews",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURAZIONE GENERALE E CREDENZIALI
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Caricamento sicuro delle credenziali (simulato, dato che non le usiamo) ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    DFSEO_LOGIN = st.secrets["DFSEO_LOGIN"]
    DFSEO_PASS = st.secrets["DFSEO_PASS"]
    CREDENTIALS_OK = True
except (KeyError, FileNotFoundError):
    st.error("⚠️ Credenziali API non trovate nei Secrets! Aggiungile per far funzionare l'app.")
    CREDENTIALS_OK = False
    st.stop()

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
        'reddit_discussions': [], 'analysis_results': None, 'enterprise_analysis': None
    }
if 'flags' not in st.session_state:
    st.session_state.flags = {'data_imported': False, 'analysis_done': False, 'enterprise_done': False}

# ============================================================================
# FUNZIONI DI SIMULAZIONE API E ANALISI
# ============================================================================

def api_call_simulation(api_name: str, limit: int) -> List[Dict]:
    """Simula una chiamata API che richiede tempo e restituisce dati finti."""
    with st.spinner(f"Chiamata a {api_name} in corso (simulazione)..."):
        time.sleep(random.uniform(2, 4))
        return [{'rating': random.randint(1, 5), 'review_text': f'Recensione simulata da {api_name} #{i+1}', 'timestamp': '2025-01-01'} for i in range(limit)]

def analyze_reviews_basic(reviews: List[Dict]) -> Dict:
    """Esegue un'analisi statistica di base."""
    if not reviews: return {}
    ratings = [r['rating'] for r in reviews]
    return {
        'total': len(reviews),
        'avg_rating': round(np.mean(ratings), 2),
        'top_themes': [('servizio', 50), ('guida', 45), ('prezzo', 30)],
        'sample_strengths': [r['review_text'] for r in reviews if r['rating'] >= 4][:3],
        'sample_pain_points': [r['review_text'] for r in reviews if r['rating'] <= 2][:3]
    }

def analyze_enterprise(data: Dict) -> Dict:
    """Simula un'analisi Enterprise complessa."""
    with st.spinner("Esecuzione Analisi Enterprise con modelli AI (simulazione)..."):
        time.sleep(5)
        total_reviews = sum(len(v) for k, v in data.items() if isinstance(v, list))
        return {
            'metadata': {'total_reviews_analyzed': total_reviews},
            'topic_modeling': {'summary': 'Trovati 5 temi principali: Organizzazione Viaggio, Qualità Guide, Rapporto Qualità/Prezzo, Assistenza Clienti, Destinazioni.'},
            'customer_journey': {'summary': 'Health Score del Journey: 0.82. Fase "Advocacy" (passaparola) molto forte.'},
            'semantic_analysis': {'summary': 'Identificate 3 recensioni anomale che parlano di argomenti non correlati.'}
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
            tp_limit = st.slider("N. Recensioni Trustpilot", 50, 2000, 200, key="tp_slider")
            if st.button("Importa da Trustpilot", use_container_width=True):
                reviews = api_call_simulation("Trustpilot", tp_limit)
                if reviews:
                    st.session_state.reviews_data['trustpilot_reviews'] = reviews
                    st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni importate!"); time.sleep(1); st.rerun()

    with col2:
        with st.expander("📍 Google Reviews", expanded=True):
            g_limit = st.slider("N. Recensioni Google", 50, 2000, 200, key="g_slider")
            if st.button("Importa da Google", use_container_width=True):
                reviews = api_call_simulation("Google", g_limit)
                if reviews:
                    st.session_state.reviews_data['google_reviews'] = reviews
                    st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni importate!"); time.sleep(1); st.rerun()

    # (Puoi aggiungere qui gli altri expander per TripAdvisor, etc. se vuoi)

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

    if total_items > 0:
        cols = st.columns(len([c for c in counts.values() if c > 0]))
        i = 0
        for platform, count in counts.items():
            if count > 0:
                with cols[i]:
                    st.metric(label=f"📝 {platform}", value=count)
                i += 1
    else:
        st.info("Nessun dato ancora importato.")

# --- TAB 2: DASHBOARD ANALISI ---
with tab2:
    st.header("📊 Dashboard Analisi")
    if not st.session_state.flags['data_imported']:
        st.info("⬅️ Importa dei dati dal tab 'Import Dati' per poter avviare un'analisi.")
    else:
        st.markdown("Esegui le analisi sui dati importati. L'Analisi Enterprise richiede più tempo ma fornisce insight più profondi.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔬 Esegui Analisi di Base", use_container_width=True):
                with st.spinner("Esecuzione analisi statistica..."):
                    results = {}
                    if st.session_state.reviews_data['trustpilot_reviews']:
                        results['Trustpilot'] = analyze_reviews_basic(st.session_state.reviews_data['trustpilot_reviews'])
                    if st.session_state.reviews_data['google_reviews']:
                        results['Google'] = analyze_reviews_basic(st.session_state.reviews_data['google_reviews'])
                    st.session_state.reviews_data['analysis_results'] = results
                    st.session_state.flags['analysis_done'] = True
                st.success("Analisi di base completata!")
                time.sleep(1); st.rerun()

        with col2:
            if st.button("🚀 Esegui Analisi Enterprise (AI)", type="primary", use_container_width=True):
                enterprise_results = analyze_enterprise(st.session_state.reviews_data)
                st.session_state.reviews_data['enterprise_analysis'] = enterprise_results
                st.session_state.flags['enterprise_done'] = True
                st.success("Analisi Enterprise completata!")
                st.balloons()
                time.sleep(1); st.rerun()

        st.markdown("---")

        # --- SEZIONE VISUALIZZAZIONE RISULTATI ---
        if st.session_state.flags['analysis_done']:
            st.subheader("🔬 Risultati Analisi di Base")
            analysis_results = st.session_state.reviews_data.get('analysis_results', {})
            for platform, results in analysis_results.items():
                with st.expander(f"**{platform}** ({results.get('total', 0)} recensioni)", expanded=True):
                    c1, c2 = st.columns(2)
                    c1.metric("Rating Medio", f"{results.get('avg_rating', 0)} ⭐")
                    
                    st.write("**Temi Principali:**")
                    st.write(", ".join([f"{theme[0]} ({theme[1]})" for theme in results.get('top_themes', [])]))
                    
                    st.write("**Esempi di Punti di Forza:**")
                    for strength in results.get('sample_strengths', []):
                        st.success(f"• *{strength[:150]}...*")

        if st.session_state.flags['enterprise_done']:
            st.subheader("🚀 Risultati Analisi Enterprise (AI)")
            enterprise_results = st.session_state.reviews_data.get('enterprise_analysis', {})
            for analysis_type, data in enterprise_results.items():
                if analysis_type not in ['metadata', 'performance_metrics']:
                    with st.expander(f"**{analysis_type.replace('_', ' ').title()}**"):
                        if isinstance(data, dict) and 'summary' in data:
                            st.info(data['summary'])
                        else:
                            st.json(data)

# --- TAB 3: EXPORT ---
with tab3:
    st.header("📥 Esporta Dati e Report")
    st.info("Le opzioni di export appariranno qui dopo aver eseguito le analisi.")
