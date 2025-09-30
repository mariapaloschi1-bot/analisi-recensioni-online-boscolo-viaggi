#!/usr/bin/env python3
"""
Reviews Analyzer v4.0 - Final Enterprise Edition by Maria
Full integration of advanced API calls, Enterprise Analysis, and SEO/Keywords Intelligence.
"""

import streamlit as st
import pandas as pd
import requests
import time
import json
import re
import numpy as np
from datetime import datetime
import logging
from openai import OpenAI
from typing import Dict, List, Optional
from collections import Counter
import io
from docx import Document

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Boscolo Viaggi Reviews",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURAZIONE INIZIALE E CREDENZIALI
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
    st.error("⚠️ Credenziali API (OPENAI_API_KEY, DFSEO_LOGIN, DFSEO_PASS) non trovate! Aggiungile nei Secrets di Streamlit Cloud per far funzionare l'app.")
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

# --- STATO DELL'APPLICAZIONE ---
if 'data' not in st.session_state:
    st.session_state.data = {
        'trustpilot': [], 'google': [], 'tripadvisor': [],
        'analysis_results': None, 'seo_analysis': None
    }
if 'flags' not in st.session_state:
    st.session_state.flags = {'data_imported': False, 'analysis_done': False}

# ============================================================================
# FUNZIONI API REALI
# ============================================================================

def post_task_and_get_id(endpoint: str, payload: List[Dict]) -> str:
    """Invia un task a DataForSEO e restituisce il task ID."""
    url = f"https://api.dataforseo.com/v3/{endpoint}"
    response = requests.post(url, auth=(DFSEO_LOGIN, DFSEO_PASS), json=payload)
    response.raise_for_status()
    data = response.json()
    
    if data["tasks_error"] > 0:
        raise Exception(f"Errore nella creazione del task: {data['tasks'][0]['status_message']}")
    
    return data["tasks"][0]["id"]

def get_task_results(endpoint: str, task_id: str) -> List[Dict]:
    """Recupera i risultati di un task da DataForSEO con polling."""
    result_url = f"https://api.dataforseo.com/v3/{endpoint}/{task_id}"
    max_attempts = 60  # Aumentato per task più lunghi
    
    for attempt in range(max_attempts):
        time.sleep(10)  # Attesa tra i tentativi
        response = requests.get(result_url, auth=(DFSEO_LOGIN, DFSEO_PASS))
        data = response.json()
        
        if data["tasks_error"] > 0:
            raise Exception(f"Errore nel recupero del task: {data['tasks'][0]['status_message']}")
        
        task = data["tasks"][0]
        if task["status_code"] == 20000: # Task completato
            if task.get("result"):
                return task["result"][0].get("items", [])
            return [] # Task completato ma senza risultati
    
    raise Exception("Timeout: il task ha impiegato troppo tempo per essere completato.")

def fetch_trustpilot_reviews(tp_url, limit):
    domain = re.search(r"/review/([^/?]+)", tp_url).group(1)
    payload = [{"domain": domain, "depth": limit, "sort_by": "recency"}]
    task_id = post_task_and_get_id("business_data/trustpilot/reviews/task_post", payload)
    return get_task_results("business_data/trustpilot/reviews/task_get", task_id)

def fetch_google_reviews(place_id, limit):
    payload = [{"place_id": place_id, "depth": limit, "sort_by": "newest", "language_name": "Italian"}]
    task_id = post_task_and_get_id("business_data/google/my_business_info/task_post", payload) # Endpoint corretto
    return get_task_results("business_data/google/reviews/task_get", task_id)

def fetch_tripadvisor_reviews(ta_url, limit):
    payload = [{"url_path": ta_url, "depth": limit, "sort_by": "newest", "language": "it"}]
    task_id = post_task_and_get_id("business_data/tripadvisor/reviews/task_post", payload)
    return get_task_results("business_data/tripadvisor/reviews/task_get", task_id)

# ============================================================================
# FUNZIONI DI ANALISI AVANZATA
# ============================================================================

def analyze_reviews_for_seo(reviews: List[Dict]) -> Dict:
    """Esegue un'analisi SEO approfondita e genera FAQ con AI."""
    with st.spinner("Esecuzione analisi SEO e generazione FAQ con AI..."):
        all_texts = [r.get('review_text', '') for r in reviews if r.get('review_text')]
        if len(all_texts) < 3: return {'error': 'Dati insufficienti per analisi SEO'}

        # Prepara contesto per OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        sample_reviews_text = "\n---\n".join([r[:300] for r in all_texts[:20]])
        
        prompt = f"""
        Sei un esperto SEO e Content Strategist. Analizza queste recensioni reali per un tour operator di nome 'Boscolo Viaggi'.
        
        RECENSIONI REALI (ESTRATTI):
        {sample_reviews_text}
        
        TASK:
        1.  **Estrai i 5 temi (N-grammi) più importanti e ricorrenti** che emergono dalle conversazioni.
        2.  **Genera 5 proposte di FAQ** basate sulle domande implicite o esplicite e sui problemi/punti di forza menzionati. Le FAQ devono essere utili per un utente che sta valutando un acquisto.
        3.  **Identifica 3 opportunità di contenuto SEO** (es. articoli di blog, pagine di destinazione) basate sui temi più caldi.
        
        Fornisci la risposta in formato JSON con questa struttura esatta:
        {{
          "top_themes": [
            {{"theme": "tema 1", "description": "Spiegazione del tema"}},
            {{"theme": "tema 2", "description": "Spiegazione del tema"}}
          ],
          "faq_proposals": [
            {{"question": "Domanda 1", "suggested_answer": "Risposta suggerita basata sulle recensioni."}},
            {{"question": "Domanda 2", "suggested_answer": "Risposta suggerita basata sulle recensioni."}}
          ],
          "content_opportunities": [
            {{"topic": "Argomento per contenuto 1", "content_type": "Tipo di contenuto (es. Articolo Blog)", "seo_value": "Valore SEO (Alto/Medio/Basso)"}},
            {{"topic": "Argomento per contenuto 2", "content_type": "Tipo di contenuto", "seo_value": "Valore SEO"}}
          ]
        }}
        """
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Sei un assistente SEO che fornisce output strutturati in JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(completion.choices[0].message.content)
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Errore parsing risposta AI per SEO: {e}")
            return {"error": "L'analisi AI non ha restituito un formato valido."}

# ============================================================================
# INTERFACCIA PRINCIPALE
# ============================================================================

st.markdown("<h1 class='main-header'>✈️ REVIEWS: Boscolo Viaggi by Maria</h1>", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🌍 Import Dati", "📊 Dashboard Analisi", "📥 Export"])

# --- TAB 1: IMPORT ---
with tab1:
    st.markdown("### 🌍 Importa Dati Reali dalle Piattaforme")
    if not CREDENTIALS_OK: st.stop()

    col1, col2 = st.columns(2)
    with col1.expander("🌟 Trustpilot", expanded=True):
        tp_url = st.text_input("URL Trustpilot", "https://it.trustpilot.com/review/boscolo.com", key="tp_url_input")
        tp_limit = st.slider("Max Recensioni TP", 50, 1000, 100, key="tp_slider")
        if st.button("Importa da Trustpilot", use_container_width=True):
            reviews = safe_api_call_with_progress(fetch_trustpilot_reviews, tp_url, tp_limit)
            if reviews:
                st.session_state.data['trustpilot'] = reviews
                st.session_state.flags['data_imported'] = True
                st.success(f"{len(reviews)} recensioni REALI importate!"); time.sleep(1); st.rerun()

    with col2.expander("✈️ TripAdvisor", expanded=True):
        ta_url = st.text_input("URL TripAdvisor", "https://www.tripadvisor.it/Attraction_Review-g187867-d24108558-Reviews-Boscolo_Viaggi-Padua_Province_of_Padua_Veneto.html", key="ta_url_input")
        ta_limit = st.slider("Max Recensioni TA", 50, 1000, 100, key="ta_slider")
        if st.button("Importa da TripAdvisor", use_container_width=True):
            reviews = safe_api_call_with_progress(fetch_tripadvisor_reviews, ta_url, ta_limit)
            if reviews:
                st.session_state.data['tripadvisor'] = reviews
                st.session_state.flags['data_imported'] = True
                st.success(f"{len(reviews)} recensioni REALI importate!"); time.sleep(1); st.rerun()

    with st.expander("📍 Google Reviews"):
        g_place_id = st.text_input("Google Place ID", "ChIJ-R_d-iV-1BIRsA7DW2s-2GA", key="g_id_input", help="Questo è il Place ID per 'Boscolo Tours S.P.A.'.")
        g_limit = st.slider("Max Recensioni Google", 50, 1000, 100, key="g_slider")
        if st.button("Importa da Google", use_container_width=True):
            reviews = safe_api_call_with_progress(fetch_google_reviews, g_place_id, g_limit)
            if reviews:
                st.session_state.data['google'] = reviews
                st.session_state.flags['data_imported'] = True
                st.success(f"{len(reviews)} recensioni REALI importate!"); time.sleep(1); st.rerun()

    st.markdown("---")
    st.subheader("Riepilogo Dati Importati")
    counts = {"Trustpilot": len(st.session_state.data['trustpilot']), "Google": len(st.session_state.data['google']), "TripAdvisor": len(st.session_state.data['tripadvisor'])}
    total_items = sum(counts.values())
    if total_items > 0:
        active_platforms = [p for p, c in counts.items() if c > 0]
        cols = st.columns(len(active_platforms))
        for i, platform in enumerate(active_platforms):
            cols[i].metric(label=f"📝 {platform}", value=counts[platform])
    else:
        st.info("Nessun dato ancora importato.")

# --- TAB 2: ANALISI ---
with tab2:
    st.header("📊 Dashboard Analisi")
    if not st.session_state.flags['data_imported']:
        st.info("⬅️ Importa dati dal tab 'Import Dati' per poter eseguire un'analisi.")
    else:
        if st.button("🚀 Esegui Analisi SEO e Generazione FAQ (AI)", type="primary", use_container_width=True):
            all_reviews = st.session_state.data['trustpilot'] + st.session_state.data['google'] + st.session_state.data['tripadvisor']
            if len(all_reviews) > 0:
                st.session_state.data['seo_analysis'] = analyze_reviews_for_seo(all_reviews)
                st.session_state.flags['analysis_done'] = True
                st.success("Analisi SEO e generazione FAQ completate!")
                st.balloons()
                time.sleep(1); st.rerun()
            else:
                st.warning("Nessuna recensione da analizzare.")

        st.markdown("---")
        
        if st.session_state.flags['analysis_done']:
            seo_results = st.session_state.data.get('seo_analysis')
            if seo_results and 'error' not in seo_results:
                st.subheader("📈 Risultati Analisi SEO & Contenuti")
                
                with st.expander("❓ **Proposte di FAQ Generate con AI**", expanded=True):
                    faqs = seo_results.get('faq_proposals', [])
                    if faqs:
                        for i, faq in enumerate(faqs, 1):
                            st.markdown(f"**Domanda {i}:** {faq['question']}")
                            st.info(f"**Risposta Suggerita:** {faq['suggested_answer']}")
                            st.markdown("---")
                
                with st.expander("💡 **Opportunità di Contenuto SEO**"):
                    opps = seo_results.get('content_opportunities', [])
                    if opps:
                        for idea in opps:
                            st.success(f"**{idea['content_type']} sul tema '{idea['topic']}'** (Valore SEO: {idea['seo_value']})")
                
                with st.expander("🔥 **Temi Principali Estratti**"):
                    themes = seo_results.get('top_themes', [])
                    if themes:
                        for theme in themes:
                            st.markdown(f"**{theme['theme'].title()}**: *{theme['description']}*")
            elif seo_results:
                st.error(f"Errore durante l'analisi SEO: {seo_results['error']}")

# --- TAB 3: EXPORT ---
with tab3:
    st.header("📥 Esporta Dati e Report")
    st.info("Funzionalità di export in costruzione.")
