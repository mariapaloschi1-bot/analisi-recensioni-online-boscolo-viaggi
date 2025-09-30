#!/usr/bin/env python3
"""
Reviews Analyzer v5.1 - Final Enterprise Edition by Maria
Full implementation of API calls, analysis dashboards, and export functionality.
"""

import streamlit as st
import pandas as pd
import requests
import time
import json
import re
import numpy as np
import logging
from openai import OpenAI
from typing import Dict, List
import threading
from docx import Document
import io

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

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    DFSEO_LOGIN = st.secrets["DFSEO_LOGIN"]
    DFSEO_PASS = st.secrets["DFSEO_PASS"]
except KeyError as e:
    st.error(f"⚠️ Manca una credenziale nei Secrets di Streamlit: {e}. L'app non può funzionare.")
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
    st.session_state.data = {'trustpilot': [], 'google': [], 'tripadvisor': [], 'analysis_results': None, 'seo_analysis': None}
if 'flags' not in st.session_state:
    st.session_state.flags = {'data_imported': False, 'analysis_done': False}

# ============================================================================
# FUNZIONI API REALI E HELPER
# ============================================================================
def safe_api_call_with_progress(api_function, *args, **kwargs):
    progress_bar = st.progress(0, text=f"Inizializzazione chiamata a {api_function.__name__}...")
    result, error = None, None
    def api_wrapper():
        nonlocal result, error
        try:
            result = api_function(*args, **kwargs)
        except Exception as e:
            error = e
    thread = threading.Thread(target=api_wrapper)
    thread.start()
    while thread.is_alive():
        progress_bar.progress(50, text="Elaborazione in corso su DataForSEO... L'operazione può richiedere diversi minuti.")
        time.sleep(5)
    thread.join()
    progress_bar.empty()
    if error: raise error
    return result

def post_task_and_get_id(endpoint: str, payload: List[Dict]) -> str:
    url = f"https://api.dataforseo.com/v3/{endpoint}"
    response = requests.post(url, auth=(DFSEO_LOGIN, DFSEO_PASS), json=payload)
    response.raise_for_status()
    data = response.json()
    if data.get("tasks_error", 1) > 0:
        msg = data['tasks'][0]['status_message']
        raise Exception(f"Errore API (Creazione Task): {msg}")
    return data["tasks"][0]["id"]

def get_task_results(endpoint: str, task_id: str) -> List[Dict]:
    result_url = f"https://api.dataforseo.com/v3/{endpoint}/task_get/{task_id}"
    for attempt in range(60):
        time.sleep(10)
        logger.info(f"Tentativo {attempt+1}/60 per il task {task_id}")
        response = requests.get(result_url, auth=(DFSEO_LOGIN, DFSEO_PASS))
        response.raise_for_status()
        data = response.json()
        task = data["tasks"][0]
        status_code = task.get("status_code")
        status_message = (task.get("status_message") or "").lower()
        if status_code == 20000:
            logger.info(f"Task {task_id} completato.")
            items = []
            if task.get("result"):
                for page in task["result"]:
                    if page and page.get("items"): items.extend(page["items"])
            return items
        elif status_code in [20100, 40602] or "queue" in status_message or "handed" in status_message:
             logger.info(f"Task {task_id} in attesa (Status: {status_message}). Continuo ad attendere.")
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
    # CORREZIONE FINALE: Usiamo location_code come richiesto dalla documentazione
    payload = [{"place_id": place_id, "limit": limit, "language_code": "it", "location_code": 2380}] # 2380 è il codice per l'Italia
    task_id = post_task_and_get_id("business_data/google/reviews/task_post", payload)
    return get_task_results("business_data/google/reviews", task_id)

def fetch_tripadvisor_reviews(ta_url, limit):
    clean_url = ta_url.split('?')[0]
    payload = [{"url": clean_url, "limit": limit}]
    task_id = post_task_and_get_id("business_data/tripadvisor/reviews/task_post", payload)
    return get_task_results("business_data/tripadvisor/reviews", task_id)

def analyze_reviews_for_seo(reviews: List[Dict]) -> Dict:
    with st.spinner("Esecuzione analisi SEO e generazione FAQ con AI..."):
        all_texts = [r.get('review_text', '') for r in reviews if r.get('review_text')]
        if len(all_texts) < 3: return {'error': 'Dati insufficienti'}
        client = OpenAI(api_key=OPENAI_API_KEY)
        sample_reviews_text = "\n---\n".join([r[:300] for r in all_texts[:20]])
        prompt = f"""Sei un esperto SEO e Content Strategist. Analizza queste recensioni reali per 'Boscolo Viaggi'.
        RECENSIONI (ESTRATTI): {sample_reviews_text}
        TASK:
        1. Estrai i 5 temi più importanti.
        2. Genera 5 proposte di FAQ basate sui temi e problemi emersi.
        3. Identifica 3 opportunità di contenuto SEO.
        Fornisci la risposta in JSON con le chiavi "top_themes", "faq_proposals", "content_opportunities".
        """
        completion = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "Sei un assistente SEO che fornisce output JSON."}, {"role": "user", "content": prompt}], response_format={"type": "json_object"})
        try:
            return json.loads(completion.choices[0].message.content)
        except (json.JSONDecodeError, IndexError):
            return {"error": "Analisi AI fallita."}

# ============================================================================
# INTERFACCIA PRINCIPALE
# ============================================================================
st.markdown("<h1 class='main-header'>✈️ REVIEWS: Boscolo Viaggi by Maria</h1>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🌍 Import Dati", "📊 Dashboard Analisi", "📥 Export"])

with tab1:
    # ... (Codice di import, identico a prima)
    st.markdown("### 🌍 Importa Dati Reali dalle Piattaforme")
    col1, col2 = st.columns(2)
    with col1.expander("🌟 Trustpilot", expanded=True):
        tp_url = st.text_input("URL Trustpilot", "https://it.trustpilot.com/review/boscolo.com", key="tp_url_input")
        tp_limit = st.slider("Max Recensioni TP", 50, 1000, 100, key="tp_slider")
        if st.button("Importa da Trustpilot", use_container_width=True):
            try:
                reviews = safe_api_call_with_progress(fetch_trustpilot_reviews, tp_url, tp_limit)
                if reviews is not None:
                    st.session_state.data['trustpilot'] = reviews
                    st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni REALI importate!"); time.sleep(2); st.rerun()
            except Exception as e:
                st.error(f"Errore Trustpilot: {e}")

    with col2.expander("✈️ TripAdvisor", expanded=True):
        ta_url = st.text_input("URL TripAdvisor", "https://www.tripadvisor.it/Attraction_Review-g187867-d24108558-Reviews-Boscolo_Viaggi-Padua_Province_of_Padua_Veneto.html", key="ta_url_input")
        ta_limit = st.slider("Max Recensioni TA", 50, 1000, 100, key="ta_slider")
        if st.button("Importa da TripAdvisor", use_container_width=True):
            try:
                reviews = safe_api_call_with_progress(fetch_tripadvisor_reviews, ta_url, ta_limit)
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
                reviews = safe_api_call_with_progress(fetch_google_reviews, g_place_id, g_limit)
                if reviews is not None:
                    st.session_state.data['google'] = reviews
                    st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni REALI importate!"); time.sleep(2); st.rerun()
            except Exception as e:
                st.error(f"Errore Google: {e}")
    
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
                st.success("Analisi SEO e generazione FAQ completate!"); st.balloons(); time.sleep(1); st.rerun()
        
        st.markdown("---")
        
        if st.session_state.flags['analysis_done']:
            seo_results = st.session_state.data.get('seo_analysis')
            if seo_results and 'error' not in seo_results:
                st.subheader("📈 Risultati Analisi SEO & Contenuti")
                
                with st.expander("❓ **Proposte di FAQ Generate con AI**", expanded=True):
                    faqs = seo_results.get('faq_proposals', [])
                    for i, faq in enumerate(faqs, 1):
                        st.markdown(f"**Domanda {i}:** {faq['question']}")
                        st.info(f"**Risposta Suggerita:** {faq['suggested_answer']}")
                        st.markdown("---")
                
                with st.expander("💡 **Opportunità di Contenuto SEO**"):
                    opps = seo_results.get('content_opportunities', [])
                    for idea in opps:
                        st.success(f"**{idea['content_type']} sul tema '{idea['topic']}'** (Valore SEO: {idea['seo_value']})")
                
                with st.expander("🔥 **Temi Principali Estratti**"):
                    themes = seo_results.get('top_themes', [])
                    for theme in themes:
                        st.markdown(f"**{theme['theme'].title()}**: *{theme['description']}*")
            elif seo_results:
                st.error(f"Errore durante l'analisi SEO: {seo_results['error']}")

with tab3:
    st.header("📥 Export Dati e Report")
    if not st.session_state.flags['analysis_done']:
        st.info("Esegui prima un'analisi per abilitare l'export.")
    else:
        st.subheader("Esporta i tuoi dati e risultati")
        
        # Esporta CSV
        all_reviews_df = pd.DataFrame(st.session_state.data['trustpilot'] + st.session_state.data['google'] + st.session_state.data['tripadvisor'])
        csv = all_reviews_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Scarica tutte le recensioni (CSV)", data=csv, file_name="reviews_export.csv", mime="text/csv", use_container_width=True)
        
        # Esporta Report Testuale
        seo_results = st.session_state.data.get('seo_analysis')
        if seo_results:
            report_text = f"Report Analisi SEO per Boscolo Viaggi - {datetime.now().strftime('%Y-%m-%d')}\n\n"
            report_text += "=== TEMI PRINCIPALI ===\n"
            for theme in seo_results.get('top_themes', []):
                report_text += f"- {theme['theme'].title()}: {theme['description']}\n"
            report_text += "\n=== FAQ SUGGERITE ===\n"
            for faq in seo_results.get('faq_proposals', []):
                report_text += f"D: {faq['question']}\nR: {faq['suggested_answer']}\n\n"
            
            st.download_button("📄 Scarica Report Analisi (TXT)", data=report_text.encode('utf-8'), file_name="seo_report.txt", mime="text/plain", use_container_width=True)
