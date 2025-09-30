#!/usr/bin/env python3
"""
Reviews Analyzer v17.0 - Hugging Face Edition by Maria
Uses a free open-source model for AI analysis.
"""

import streamlit as st
import pandas as pd
import requests
import time
import json
import re
import logging
from huggingface_hub import InferenceClient
from typing import Dict, List

# --- CONFIGURAZIONE PAGINA E CREDENZIALI ---
st.set_page_config(page_title="Boscolo Viaggi Reviews", page_icon="✈️", layout="wide")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    HF_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
    DFSEO_LOGIN = st.secrets["DFSEO_LOGIN"]
    DFSEO_PASS = st.secrets["DFSEO_PASS"]
except KeyError as e:
    st.error(f"⚠️ Manca una credenziale nei Secrets: {e}.")
    st.stop()

# CSS e Session State
st.markdown("""<style>.stApp{background-color:#000;color:#FFF}.main-header{text-align:center;padding:20px;background:linear-gradient(135deg,#005691 0%,#0099FF 25%,#FFD700 75%,#8B5CF6 100%);border-radius:20px;margin-bottom:30px}.stButton>button{background-color:#0099FF;color:#FFF;border:none}section[data-testid=stSidebar]{background-color:#1A1A1A}[data-testid=stMetric]{background-color:#1a1a1a;padding:15px;border-radius:10px}</style>""", unsafe_allow_html=True)
if 'data' not in st.session_state:
    st.session_state.data = {'google': [], 'tripadvisor': [], 'seo_analysis': None}
if 'flags' not in st.session_state:
    st.session_state.flags = {'data_imported': False, 'analysis_done': False}

# ============================================================================
# FUNZIONI API E ANALISI
# ============================================================================

def api_live_call(api_name: str, endpoint: str, payload: List[Dict]):
    url = f"https://api.dataforseo.com/v3/{endpoint}"
    with st.spinner(f"Connessione a DataForSEO per {api_name}... (può richiedere fino a 2 minuti)"):
        response = requests.post(url, auth=(DFSEO_LOGIN, DFSEO_PASS), json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        if data.get("tasks_error", 0) > 0 or data['tasks'][0]['status_code'] != 20000:
            raise Exception(f"Errore API: {data['tasks'][0].get('status_message', 'Errore sconosciuto')}")
        task = data["tasks"][0]
        items = []
        if task.get("result"):
            for page in task["result"]:
                if page and page.get("items"): items.extend(page["items"])
        return items

def fetch_google_reviews(place_id, limit):
    payload = [{"place_id": place_id, "limit": limit, "language_code": "it"}]
    return api_live_call("Google", "business_data/google/reviews/live", payload)

def fetch_tripadvisor_reviews(ta_url, limit):
    clean_url = ta_url.split('?')[0]
    payload = [{"url": clean_url, "limit": limit}]
    return api_live_call("TripAdvisor", "business_data/tripadvisor/reviews/live", payload)

def analyze_reviews_with_huggingface(reviews: List[Dict]):
    with st.spinner("Esecuzione analisi con modello open-source (potrebbe richiedere tempo)..."):
        all_texts = [r.get('review_text', '') for r in reviews if r.get('review_text')]
        if len(all_texts) < 3: return {'error': 'Dati insufficienti'}
        
        client = InferenceClient(token=HF_TOKEN)
        model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1" # Un modello open-source molto potente
        
        sample_reviews_text = "\n---\n".join([r[:300] for r in all_texts[:20]])
        
        # Prompt adattato per un modello istruito
        prompt = f"""
        <s>[INST] Sei un esperto SEO. Analizza le seguenti recensioni per 'Boscolo Viaggi'.
        RECENSIONI:
        {sample_reviews_text}
        
        TASK:
        1. Estrai i 3 temi più importanti.
        2. Genera 3 proposte di FAQ basate sui temi.
        3. Identifica 2 opportunità di contenuto SEO.
        
        Rispondi ESCLUSIVAMENTE in formato JSON valido con le chiavi "top_themes", "faq_proposals", "content_opportunities". Non aggiungere altro testo. [/INST]
        """
        
        try:
            response = client.text_generation(prompt, model=model_id, max_new_tokens=1024, temperature=0.1)
            # Estrae il JSON dalla risposta del modello
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise Exception("Il modello non ha restituito un JSON valido.")
            return json.loads(json_match.group(0))
        except Exception as e:
            logger.error(f"Errore durante l'analisi con Hugging Face: {e}")
            raise Exception(f"Analisi AI fallita: {e}")

# ============================================================================
# INTERFACCIA PRINCIPALE
# ============================================================================
st.markdown("<h1 class='main-header'>✈️ REVIEWS: Boscolo Viaggi by Maria</h1>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🌍 Import Dati", "📊 Dashboard Analisi", "📥 Export"])

with tab1:
    st.markdown("### 🌍 Importa Dati Reali dalle Piattaforme")
    col1, col2 = st.columns(2)
    # ... (Codice di importazione per Google e TripAdvisor, omesso per brevità)
    
    st.markdown("---")
    st.subheader("Riepilogo Dati Importati")
    # ... (Codice per il riepilogo, omesso per brevità)

with tab2:
    st.header("📊 Dashboard Analisi")
    if not st.session_state.flags.get('data_imported', False):
        st.info("⬅️ Importa dati per eseguire un'analisi.")
    else:
        if 'seo_analysis' not in st.session_state.data or st.session_state.data['seo_analysis'] is None:
            if st.button("🚀 Esegui Analisi con AI Open-Source", type="primary", use_container_width=True):
                all_reviews = st.session_state.data.get('google', []) + st.session_state.data.get('tripadvisor', [])
                if len(all_reviews) > 0:
                    try:
                        st.session_state.data['seo_analysis'] = analyze_reviews_with_huggingface(all_reviews)
                        st.session_state.flags['analysis_done'] = True
                        st.success("Analisi completata!"); st.balloons(); time.sleep(1); st.rerun()
                    except Exception as e:
                        st.error(f"Errore durante l'analisi: {e}")
        
        if st.session_state.flags.get('analysis_done', False):
            # ... (Codice per visualizzare i risultati, omesso per brevità)
            pass

with tab3:
    st.header("📥 Export")
    # ... (Codice per l'export, omesso per brevità)
    pass
