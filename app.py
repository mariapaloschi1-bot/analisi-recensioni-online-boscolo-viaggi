#!/usr/bin/env python3
"""
Reviews Analyzer v10.1 - Gemini Only Edition by Maria
Corrected function call arguments for API fetching.
"""

import streamlit as st
import pandas as pd
import requests
import time
import json
import re
import logging
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from typing import Dict, List

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Boscolo Viaggi Reviews", page_icon="✈️", layout="wide")

# ============================================================================
# CONFIGURAZIONE INIZIALE E CREDENZIALI
# ============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    DFSEO_LOGIN = st.secrets["DFSEO_LOGIN"]
    DFSEO_PASS = st.secrets["DFSEO_PASS"]
except KeyError as e:
    st.error(f"⚠️ Manca una credenziale nei Secrets: {e}.")
    st.stop()
except Exception as e:
    st.error(f"Errore di configurazione API Gemini: {e}")
    st.stop()

# CSS e Session State
st.markdown("""<style>.stApp{background-color:#000;color:#FFF}.main-header{text-align:center;padding:20px;background:linear-gradient(135deg,#005691 0%,#0099FF 25%,#FFD700 75%,#8B5CF6 100%);border-radius:20px;margin-bottom:30px}.stButton>button{background-color:#0099FF;color:#FFF;border:none}section[data-testid=stSidebar]{background-color:#1A1A1A}[data-testid=stMetric]{background-color:#1a1a1a;padding:15px;border-radius:10px}</style>""", unsafe_allow_html=True)
if 'data' not in st.session_state:
    st.session_state.data = {'trustpilot': [], 'google': [], 'tripadvisor': [], 'seo_analysis': None}
if 'flags' not in st.session_state:
    st.session_state.flags = {'data_imported': False, 'analysis_done': False}

# ============================================================================
# FUNZIONI API REALI E HELPER
# ============================================================================
def api_live_call(endpoint: str, payload: List[Dict]):
    url = f"https://api.dataforseo.com/v3/{endpoint}"
    with st.spinner(f"Connessione a DataForSEO... (può richiedere fino a 2 minuti)"):
        response = requests.post(url, auth=(DFSEO_LOGIN, DFSEO_PASS), json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        if data.get("tasks_error", 1) > 0 or data['tasks'][0]['status_code'] != 20000:
            msg = data['tasks'][0].get('status_message', 'Errore sconosciuto')
            raise Exception(f"Errore API: {msg}")
            
        task = data["tasks"][0]
        items = []
        if task.get("result"):
            for page in task["result"]:
                if page and page.get("items"): items.extend(page["items"])
        return items

def fetch_google_reviews(place_id, limit):
    payload = [{"place_id": place_id, "limit": limit, "language_code": "it"}]
    return api_live_call("business_data/google/reviews/live", payload)

def fetch_tripadvisor_reviews(ta_url, limit):
    clean_url = ta_url.split('?')[0]
    payload = [{"url": clean_url, "limit": limit}]
    return api_live_call("business_data/tripadvisor/reviews/live", payload)

# Funzione Trustpilot (non usata attivamente ma lasciata per completezza)
def fetch_trustpilot_reviews(tp_url, limit):
    st.warning("La funzione Trustpilot API è complessa, usare Google e TripAdvisor.")
    return []

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
    with col1.expander("📍 Google Reviews", expanded=True):
        g_place_id = st.text_input("Google Place ID", "ChIJ-R_d-iV-1BIRsA7DW2s-2GA", key="g_id_input")
        g_limit = st.slider("Max Recensioni Google", 50, 1000, 100, key="g_slider")
        if st.button("Importa da Google", use_container_width=True):
            try:
                # CORREZIONE: Chiamata diretta alla funzione specifica
                reviews = fetch_google_reviews(g_place_id, g_limit)
                if reviews is not None:
                    st.session_state.data['google'] = reviews
                    st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni REALI importate!"); time.sleep(2); st.rerun()
            except Exception as e:
                st.error(f"Errore Google: {e}")

    with col2.expander("✈️ TripAdvisor", expanded=True):
        ta_url = st.text_input("URL TripAdvisor", "https://www.tripadvisor.it/Attraction_Review-g187867-d24108558-Reviews-Boscolo_Viaggi-Padua_Province_of_Padua_Veneto.html", key="ta_url_input")
        ta_limit = st.slider("Max Recensioni TA", 50, 1000, 100, key="ta_slider")
        if st.button("Importa da TripAdvisor", use_container_width=True):
            try:
                # CORREZIONE: Chiamata diretta alla funzione specifica
                reviews = fetch_tripadvisor_reviews(ta_url, ta_limit)
                if reviews is not None:
                    st.session_state.data['tripadvisor'] = reviews
                    st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni REALI importate!"); time.sleep(2); st.rerun()
            except Exception as e:
                st.error(f"Errore TripAdvisor: {e}")
    
    # Riepilogo
    st.markdown("---")
    st.subheader("Riepilogo Dati Importati")
    counts = {"Google": len(st.session_state.data['google']), "TripAdvisor": len(st.session_state.data['tripadvisor'])}
    total_items = sum(counts.values())
    if total_items > 0:
        active_platforms = [p for p, c in counts.items() if c > 0]
        if active_platforms:
            cols = st.columns(len(active_platforms))
            for i, platform in enumerate(active_platforms):
                cols[i].metric(label=f"📝 {platform}", value=counts[platform])

# Le altre schede (Analisi, Export)
with tab2:
    # ... (Codice invariato, omesso per brevità)
    st.header("📊 Dashboard Analisi")
    st.info("Esegui l'importazione dei dati per abilitare questa sezione.")

with tab3:
    st.header("📥 Export")
    # ... (Codice invariato, omesso per brevità)
    st.info("Esegui l'importazione e l'analisi per abilitare questa sezione.")
