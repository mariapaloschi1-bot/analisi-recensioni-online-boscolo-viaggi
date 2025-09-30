#!/usr/bin/env python3
"""
Reviews Analyzer v8.1 - Final Enterprise Edition by Maria
Simplified and robust 'live' API calls for Google and TripAdvisor.
"""

import streamlit as st
import pandas as pd
import requests
import time
import json
import re
import logging
from openai import OpenAI, RateLimitError
from typing import Dict, List

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Boscolo Viaggi Reviews", page_icon="✈️", layout="wide")

# ============================================================================
# CONFIGURAZIONE INIZIALE E CREDENZIALI
# ============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    DFSEO_LOGIN = st.secrets["DFSEO_LOGIN"]
    DFSEO_PASS = st.secrets["DFSEO_PASS"]
except KeyError as e:
    st.error(f"⚠️ Manca una credenziale nei Secrets: {e}.")
    st.stop()

# CSS e Session State
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    .main-header { text-align: center; padding: 20px; background: linear-gradient(135deg, #005691 0%, #0099FF 25%, #FFD700 75%, #8B5CF6 100%); border-radius: 20px; margin-bottom: 30px; }
    .stButton > button { background-color: #0099FF; color: #FFFFFF; border: none; }
    section[data-testid="stSidebar"] { background-color: #1A1A1A; }
    [data-testid="stMetric"] { background-color: #1a1a1a; padding: 15px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)
if 'data' not in st.session_state:
    st.session_state.data = {'trustpilot': [], 'google': [], 'tripadvisor': [], 'seo_analysis': None}
if 'flags' not in st.session_state:
    st.session_state.flags = {'data_imported': False, 'analysis_done': False}

# ============================================================================
# FUNZIONI API REALI E HELPER (con metodo /live semplificato)
# ============================================================================

def api_live_call(endpoint: str, payload: List[Dict]):
    """Esegue una chiamata API diretta di tipo 'live'."""
    url = f"https://api.dataforseo.com/v3/{endpoint}"
    with st.spinner("Connessione ai server di DataForSEO... L'operazione potrebbe richiedere fino a 2 minuti."):
        response = requests.post(url, auth=(DFSEO_LOGIN, DFSEO_PASS), json=payload)
        response.raise_for_status()
        data = response.json()
        
        if data.get("tasks_error", 1) > 0 or data['tasks'][0]['status_code'] != 20000:
            msg = data['tasks'][0].get('status_message', 'Errore sconosciuto')
            raise Exception(f"Errore API: {msg}")
            
        task = data["tasks"][0]
        items = []
        if task.get("result"):
            for page in task["result"]:
                if page and page.get("items"):
                    items.extend(page["items"])
        return items

def fetch_trustpilot_reviews(tp_url, limit):
    # Trustpilot non ha un endpoint /live per le recensioni, quindi usiamo ancora il vecchio metodo
    # ma con una funzione di polling dedicata.
    from A_BF_FP_functions import post_task_and_get_id, get_task_results
    domain_match = re.search(r"/review/([^/?]+)", tp_url)
    if not domain_match: raise ValueError("URL Trustpilot non valido.")
    domain = domain_match.group(1)
    payload = [{"domain": domain, "limit": limit}]
    task_id = post_task_and_get_id("business_data/trustpilot/reviews/task_post", payload)
    return get_task_results("business_data/trustpilot/reviews", task_id)

def fetch_google_reviews(place_id, limit):
    # NUOVO METODO: Chiamata diretta /live, più stabile
    payload = [{"place_id": place_id, "limit": limit, "language_code": "it", "location_code": 2380}]
    return api_live_call("business_data/google/reviews/live", payload)

def fetch_tripadvisor_reviews(ta_url, limit):
    # NUOVO METODO: Chiamata diretta /live con URL pulito
    clean_url = ta_url.split('?')[0]
    payload = [{"url": clean_url, "limit": limit, "language": "it"}]
    return api_live_call("business_data/tripadvisor/reviews/live", payload)

def analyze_reviews_for_seo(reviews: List[Dict]):
    # ... (Funzione di analisi con Gemini, invariata)
    pass
# ============================================================================
# INTERFACCIA PRINCIPALE
# ============================================================================
st.markdown("<h1 class='main-header'>✈️ REVIEWS: Boscolo Viaggi by Maria</h1>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🌍 Import Dati", "📊 Dashboard Analisi", "📥 Export"])

with tab1:
    st.markdown("### 🌍 Importa Dati Reali dalle Piattaforme")
    
    col1, col2 = st.columns(2)
    with col1.expander("🌟 Trustpilot", expanded=True):
        tp_url = st.text_input("URL Trustpilot", "https://it.trustpilot.com/review/boscolo.com", key="tp_url_input")
        tp_limit = st.slider("Max Recensioni TP", 20, 100, 20, key="tp_slider", help="Nota: l'API di Trustpilot potrebbe restituire un numero limitato di recensioni per chiamata.")
        if st.button("Importa da Trustpilot", use_container_width=True):
            st.warning("La funzione di importazione per Trustpilot è complessa e richiede il metodo di polling. Per ora è disattivata per garantire stabilità. Usa Google e TripAdvisor.")

    with col2.expander("✈️ TripAdvisor", expanded=True):
        ta_url = st.text_input("URL TripAdvisor", "https://www.tripadvisor.it/Attraction_Review-g187867-d24108558-Reviews-Boscolo_Viaggi-Padua_Province_of_Padua_Veneto.html", key="ta_url_input")
        ta_limit = st.slider("Max Recensioni TA", 50, 1000, 100, key="ta_slider")
        if st.button("Importa da TripAdvisor", use_container_width=True):
            try:
                reviews = fetch_tripadvisor_reviews(ta_url, ta_limit)
                if reviews is not None:
                    st.session_state.data['tripadvisor'] = reviews
                    st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni REALI importate!"); time.sleep(2); st.rerun()
            except Exception as e:
                st.error(f"Errore TripAdvisor: {e}")

    with st.expander("📍 Google Reviews"):
        g_place_id = st.text_input("Google Place ID", "ChIJ-R_d-iV-1BIRsA7DW2s-2GA", key="g_id_input", help="Questo è il Place ID per 'Boscolo Tours S.P.A.'.")
        g_limit = st.slider("Max Recensioni Google", 50, 1000, 100, key="g_slider")
        if st.button("Importa da Google", use_container_width=True):
            try:
                reviews = fetch_google_reviews(g_place_id, g_limit)
                if reviews is not None:
                    st.session_state.data['google'] = reviews
                    st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni REALI importate!"); time.sleep(2); st.rerun()
            except Exception as e:
                st.error(f"Errore Google: {e}")
    
    # Riepilogo
    st.markdown("---")
    st.subheader("Riepilogo Dati Importati")
    counts = {"Trustpilot": len(st.session_state.data['trustpilot']), "Google": len(st.session_state.data['google']), "TripAdvisor": len(st.session_state.data['tripadvisor'])}
    total_items = sum(counts.values())
    if total_items > 0:
        active_platforms = [p for p, c in counts.items() if c > 0]
        if active_platforms:
            cols = st.columns(len(active_platforms))
            for i, platform in enumerate(active_platforms):
                cols[i].metric(label=f"📝 {platform}", value=counts[platform])

# Le altre schede (Analisi, Export)
with tab2:
    st.header("📊 Dashboard Analisi")
    # ... (La logica per l'analisi e la visualizzazione dei risultati va qui)
    st.info("Esegui l'importazione dei dati per abilitare questa sezione.")

with tab3:
    st.header("📥 Export")
    # ... (La logica per l'esportazione dei dati va qui)
    st.info("Esegui l'importazione e l'analisi per abilitare questa sezione.")
