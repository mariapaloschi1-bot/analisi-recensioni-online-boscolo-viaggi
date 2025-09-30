#!/usr/bin/env python3
"""
Reviews Analyzer v5.2 - Final Enterprise Edition by Maria
Robust error handling for OpenAI Rate Limits and TripAdvisor API payloads.
"""

import streamlit as st
import pandas as pd
import requests
import time
import json
import re
import logging
from openai import OpenAI
from typing import Dict, List

# --- CONFIGURAZIONE PAGINA E CREDENZIALI ---
st.set_page_config(page_title="Boscolo Viaggi Reviews", page_icon="✈️", layout="wide")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    DFSEO_LOGIN = st.secrets["DFSEO_LOGIN"]
    DFSEO_PASS = st.secrets["DFSEO_PASS"]
except KeyError as e:
    st.error(f"⚠️ Manca una credenziale nei Secrets: {e}.")
    st.stop()

# --- CSS E SESSION STATE ---
st.markdown("""<style>/* ... CSS omesso per brevità ... */</style>""", unsafe_allow_html=True)
if 'data' not in st.session_state:
    st.session_state.data = {'trustpilot': [], 'google': [], 'tripadvisor': [], 'seo_analysis': None}
if 'flags' not in st.session_state:
    st.session_state.flags = {'data_imported': False, 'analysis_done': False}

# ============================================================================
# FUNZIONI API E HELPER
# ============================================================================

def post_task_and_get_id(endpoint: str, payload: List[Dict]) -> str:
    # ... (Funzione identica a prima)
    url = f"https://api.dataforseo.com/v3/{endpoint}"
    response = requests.post(url, auth=(DFSEO_LOGIN, DFSEO_PASS), json=payload)
    response.raise_for_status()
    data = response.json()
    if data.get("tasks_error", 1) > 0: raise Exception(f"Errore API (Creazione Task): {data['tasks'][0]['status_message']}")
    return data["tasks"][0]["id"]

def get_task_results(endpoint: str, task_id: str) -> List[Dict]:
    # ... (Funzione identica a prima)
    result_url = f"https://api.dataforseo.com/v3/{endpoint}/task_get/{task_id}"
    for attempt in range(60):
        time.sleep(10)
        response = requests.get(result_url, auth=(DFSEO_LOGIN, DFSEO_PASS))
        response.raise_for_status()
        data = response.json()
        task = data["tasks"][0]
        status_code = task.get("status_code")
        status_message = (task.get("status_message") or "").lower()
        if status_code == 20000:
            items = []
            if task.get("result"):
                for page in task["result"]:
                    if page and page.get("items"): items.extend(page["items"])
            return items
        elif status_code in [20100, 40602] or "queue" in status_message or "handed" in status_message:
             continue
        else:
            raise Exception(f"Stato task non valido: {status_code} - {task.get('status_message')}")
    raise Exception("Timeout: il task ha impiegato troppo tempo.")

def fetch_trustpilot_reviews(tp_url, limit):
    domain_match = re.search(r"/review/([^/?]+)", tp_url)
    if not domain_match: raise ValueError("URL Trustpilot non valido.")
    domain = domain_match.group(1)
    payload = [{"domain": domain, "limit": limit}]
    task_id = post_task_and_get_id("business_data/trustpilot/reviews/task_post", payload)
    return get_task_results("business_data/trustpilot/reviews", task_id)

def fetch_google_reviews(place_id, limit):
    payload = [{"place_id": place_id, "limit": limit, "language_code": "it", "location_code": 2380}]
    task_id = post_task_and_get_id("business_data/google/reviews/task_post", payload)
    return get_task_results("business_data/google/reviews", task_id)

def fetch_tripadvisor_reviews(ta_url, limit):
    # CORREZIONE FINALE PER TRIPADVISOR: Estrae i codici g e d dall'URL
    match = re.search(r"-g(\d+)-d(\d+)-", ta_url)
    if not match:
        raise ValueError("URL TripAdvisor non valido o in formato non supportato. Deve contenere i codici '-g' e '-d'.")
    
    location_id = int(match.group(1))
    entity_id = match.group(2)
    
    payload = [{"location_id": location_id, "entity_id": entity_id, "limit": limit, "language": "it"}]
    task_id = post_task_and_get_id("business_data/tripadvisor/reviews/task_post", payload)
    return get_task_results("business_data/tripadvisor/reviews", task_id)

def analyze_reviews_for_seo(reviews: List[Dict]):
    # ... (Funzione identica a prima)
    with st.spinner("Esecuzione analisi SEO e generazione FAQ con AI..."):
        all_texts = [r.get('review_text', '') for r in reviews if r.get('review_text')]
        if len(all_texts) < 3: return {'error': 'Dati insufficienti'}
        client = OpenAI(api_key=OPENAI_API_KEY)
        sample_reviews_text = "\n---\n".join([r[:300] for r in all_texts[:20]])
        prompt = f"""Sei un esperto SEO. Analizza queste recensioni per 'Boscolo Viaggi'.
        RECENSIONI: {sample_reviews_text}
        TASK:
        1. Estrai i 5 temi più importanti.
        2. Genera 5 proposte di FAQ.
        3. Identifica 3 opportunità di contenuto SEO.
        Rispondi in JSON con le chiavi "top_themes", "faq_proposals", "content_opportunities".
        """
        completion = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "Sei un assistente SEO che fornisce output JSON."}, {"role": "user", "content": prompt}], response_format={"type": "json_object"})
        try: return json.loads(completion.choices[0].message.content)
        except (json.JSONDecodeError, IndexError): return {"error": "Analisi AI fallita."}

# ============================================================================
# INTERFACCIA PRINCIPALE
# ============================================================================

st.markdown("<h1 class='main-header'>✈️ REVIEWS: Boscolo Viaggi by Maria</h1>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🌍 Import Dati", "📊 Dashboard Analisi", "📥 Export"])

# --- TAB 1: IMPORT ---
with tab1:
    # ... (Codice di import identico, ma ora chiama la nuova funzione per TripAdvisor)
    st.markdown("### 🌍 Importa Dati Reali dalle Piattaforme")
    col1, col2 = st.columns(2)
    with col1.expander("🌟 Trustpilot", expanded=True):
        tp_url = st.text_input("URL Trustpilot", "https://it.trustpilot.com/review/boscolo.com", key="tp_url_input")
        tp_limit = st.slider("Max Recensioni TP", 50, 1000, 100, key="tp_slider")
        if st.button("Importa da Trustpilot", use_container_width=True):
            try:
                # ...
            except Exception as e:
                st.error(f"Errore Trustpilot: {e}")

    with col2.expander("✈️ TripAdvisor", expanded=True):
        ta_url = st.text_input("URL TripAdvisor", "https://www.tripadvisor.it/Attraction_Review-g187867-d24108558-Reviews-Boscolo_Viaggi-Padua_Province_of_Padua_Veneto.html", key="ta_url_input")
        ta_limit = st.slider("Max Recensioni TA", 50, 1000, 100, key="ta_slider")
        if st.button("Importa da TripAdvisor", use_container_width=True):
            try:
                reviews = safe_api_call_with_progress(fetch_tripadvisor_reviews, ta_url, ta_limit)
                # ...
            except Exception as e:
                st.error(f"Errore TripAdvisor: {e}")
    # ... (resto della UI di import)


# --- TAB 2: ANALISI ---
with tab2:
    st.header("📊 Dashboard Analisi")
    if not st.session_state.flags.get('data_imported', False):
        st.info("⬅️ Importa dati dal tab 'Import Dati' per poter eseguire un'analisi.")
    else:
        # CORREZIONE PER OpenAI RateLimit: non mostrare il pulsante se l'analisi è già stata fatta
        if st.session_state.data.get('seo_analysis') is None:
            if st.button("🚀 Esegui Analisi SEO e Generazione FAQ (AI)", type="primary", use_container_width=True):
                all_reviews = st.session_state.data['trustpilot'] + st.session_state.data['google'] + st.session_state.data['tripadvisor']
                if len(all_reviews) > 0:
                    st.session_state.data['seo_analysis'] = analyze_reviews_for_seo(all_reviews)
                    st.session_state.flags['analysis_done'] = True
                    st.success("Analisi SEO e generazione FAQ completate!"); st.balloons(); time.sleep(1); st.rerun()
        
        st.markdown("---")
        
        if st.session_state.flags.get('analysis_done', False):
            # ... (UI per mostrare i risultati, identica a prima)
            pass

# --- TAB 3: EXPORT ---
with tab3:
    st.header("📥 Export")
    # ... (UI per l'export, identica a prima)
    pass
