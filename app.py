#!/usr/bin/env python3
"""
Reviews Analyzer v18.0 - Final, Complete, and Merged Edition by Maria
Full restoration of the original, robust, task-based API fetching logic for all platforms.
All analysis and export tabs are fully functional, using Gemini as the primary AI.
"""

import streamlit as st
import pandas as pd
import requests
import time
import json
import re
import logging
import google.generativeai as genai
from openai import RateLimitError
from google.api_core import exceptions as google_exceptions
from typing import Dict, List
import threading
import io

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Boscolo Viaggi Reviews", page_icon="✈️", layout="wide")

# ============================================================================
# CONFIGURAZIONE INIZIALE E CREDENZIALI
# ============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    DFSEO_LOGIN = st.secrets["DFSEO_LOGIN"]
    DFSEO_PASS = st.secrets["DFSEO_PASS"]
    genai.configure(api_key=GEMINI_API_KEY)
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
# FUNZIONI API REALI E HELPER (Metodo Task-Based Originale)
# ============================================================================

def safe_api_call_with_progress(api_function, *args, **kwargs):
    progress_bar = st.progress(0, text=f"Inizializzazione chiamata a {api_function.__name__}...")
    result, error = None, None
    def api_wrapper():
        nonlocal result, error
        try: result = api_function(*args, **kwargs)
        except Exception as e: error = e
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
    if data.get("tasks_error", 0) > 0 or data['tasks'][0]['status_code'] not in [20000, 20100]:
        msg = data['tasks'][0].get('status_message', 'Errore sconosciuto')
        raise Exception(f"Errore API (Creazione Task): {msg}")
    return data["tasks"][0]["id"]

def get_task_results(endpoint: str, task_id: str) -> List[Dict]:
    result_url = f"https://api.dataforseo.com/v3/{endpoint}/task_get/{task_id}"
    for attempt in range(90): # Tenta per 15 minuti
        time.sleep(10)
        logger.info(f"Tentativo {attempt+1}/90 per il task {task_id}")
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
                    if page and page.get("items") is not None: items.extend(page["items"])
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
    payload = [{"domain": domain, "depth": limit, "sort_by": "recency"}]
    task_id = post_task_and_get_id("business_data/trustpilot/reviews/task_post", payload)
    return get_task_results("business_data/trustpilot/reviews", task_id)

def fetch_google_reviews(place_id, limit):
    payload = [{"place_id": place_id, "depth": limit, "sort_by": "newest", "language_name": "Italian", "location_name": "Italy"}]
    task_id = post_task_and_get_id("business_data/google/reviews/task_post", payload)
    return get_task_results("business_data/google/reviews", task_id)

def fetch_tripadvisor_reviews(ta_url, limit):
    payload = [{"url_path": ta_url.split('?')[0], "depth": limit, "language_name": "Italian", "location_name": "Italy"}]
    task_id = post_task_and_get_id("business_data/tripadvisor/reviews/task_post", payload)
    return get_task_results("business_data/tripadvisor/reviews", task_id)

@st.cache_resource
def get_available_gemini_model():
    preferred_models = ['models/gemini-1.5-flash-latest', 'models/gemini-pro', 'models/gemini-1.0-pro']
    for model_name in preferred_models:
        try:
            model = genai.get_model(model_name)
            if 'generateContent' in model.supported_generation_methods:
                logger.info(f"Modello Gemini disponibile trovato: {model_name}")
                return model_name
        except Exception:
            logger.warning(f"Modello {model_name} non trovato.")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            logger.info(f"Trovato modello di fallback: {m.name}")
            return m.name
    raise Exception("Nessun modello Gemini compatibile trovato per il tuo account.")

def analyze_reviews_for_seo(reviews: List[Dict]):
    with st.spinner("Esecuzione analisi SEO e generazione FAQ con Gemini..."):
        all_texts = [r.get('review_text', '') for r in reviews if r.get('review_text')]
        if len(all_texts) < 3: return {'error': 'Dati insufficienti'}
        
        model_name = get_available_gemini_model()
        model = genai.GenerativeModel(model_name)
        
        sample_reviews_text = "\n---\n".join([r[:300] for r in all_texts[:20]])
        
        prompt = f"""Sei un esperto SEO. Analizza queste recensioni per 'Boscolo Viaggi'.
        RECENSIONI: {sample_reviews_text}
        TASK:
        1. Estrai i 5 temi più importanti.
        2. Genera 5 proposte di FAQ.
        3. Identifica 3 opportunità di contenuto SEO.
        Rispondi in formato JSON valido con le chiavi "top_themes", "faq_proposals", "content_opportunities".
        """
        try:
            response = model.generate_content(prompt)
            cleaned_response = re.search(r'```json\n(.*?)\n```', response.text, re.DOTALL)
            json_text = cleaned_response.group(1) if cleaned_response else response.text
            return json.loads(json_text)
        except (google_exceptions.ResourceExhausted, google_exceptions.InternalServerError) as e:
            raise Exception("ERRORE GEMINI: Limiti di utilizzo superati. Controlla il tuo account Google AI Studio.")
        except Exception as e:
            raise Exception(f"Analisi AI con Gemini fallita: {e}")

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
        tp_limit = st.slider("Max Recensioni TP", 50, 1000, 100, key="tp_slider")
        if st.button("Importa da Trustpilot", use_container_width=True):
            try:
                reviews = safe_api_call_with_progress(fetch_trustpilot_reviews, tp_url, tp_limit)
                if reviews is not None:
                    st.session_state.data['trustpilot'] = reviews; st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni importate!"); time.sleep(1); st.rerun()
            except Exception as e: st.error(f"Errore Trustpilot: {e}")
            
    with col2.expander("✈️ TripAdvisor", expanded=True):
        ta_url = st.text_input("URL TripAdvisor", "https://www.tripadvisor.it/Attraction_Review-g187867-d24108558-Reviews-Boscolo_Viaggi-Padua_Province_of_Padua_Veneto.html", key="ta_url_input")
        ta_limit = st.slider("Max Recensioni TA", 50, 1000, 100, key="ta_slider")
        if st.button("Importa da TripAdvisor", use_container_width=True):
            try:
                reviews = safe_api_call_with_progress(fetch_tripadvisor_reviews, ta_url, ta_limit)
                if reviews is not None:
                    st.session_state.data['tripadvisor'] = reviews; st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni importate!"); time.sleep(1); st.rerun()
            except Exception as e: st.error(f"Errore TripAdvisor: {e}")

    with st.expander("📍 Google Reviews"):
        g_place_id = st.text_input("Google Place ID", "ChIJ-R_d-iV-1BIRsA7DW2s-2GA", key="g_id_input")
        g_limit = st.slider("Max Recensioni Google", 50, 1000, 100, key="g_slider")
        if st.button("Importa da Google", use_container_width=True):
            try:
                reviews = safe_api_call_with_progress(fetch_google_reviews, g_place_id, g_limit)
                if reviews is not None:
                    st.session_state.data['google'] = reviews; st.session_state.flags['data_imported'] = True
                    st.success(f"{len(reviews)} recensioni importate!"); time.sleep(1); st.rerun()
            except Exception as e: st.error(f"Errore Google: {e}")
    
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
        if 'seo_analysis' not in st.session_state.data or st.session_state.data['seo_analysis'] is None:
            if st.button("🚀 Esegui Analisi SEO e Generazione FAQ (con Gemini)", type="primary", use_container_width=True):
                all_reviews = st.session_state.data['trustpilot'] + st.session_state.data['google'] + st.session_state.data['tripadvisor']
                if len(all_reviews) > 0:
                    try:
                        st.session_state.data['seo_analysis'] = analyze_reviews_for_seo(all_reviews)
                        st.session_state.flags['analysis_done'] = True
                        st.success("Analisi completata!"); st.balloons(); time.sleep(1); st.rerun()
                    except Exception as e:
                        st.error(f"Si è verificato un errore durante l'analisi: {e}")
        
        if st.session_state.flags.get('analysis_done', False):
            st.markdown("---")
            seo_results = st.session_state.data.get('seo_analysis')
            if seo_results and 'error' not in seo_results:
                st.subheader("📈 Risultati Analisi SEO & Contenuti (generati da Gemini)")
                
                with st.expander("❓ **Proposte di FAQ Generate con AI**", expanded=True):
                    faqs = seo_results.get('faq_proposals', [])
                    if faqs:
                        for i, faq in enumerate(faqs, 1):
                            st.markdown(f"**Domanda {i}:** {faq['question']}")
                            st.info(f"**Risposta Suggerita:** {faq['suggested_answer']}")
                
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

with tab3:
    st.header("📥 Export")
    if not st.session_state.flags['data_imported']:
        st.info("Importa dei dati per abilitare l'export.")
    else:
        st.subheader("Esporta i tuoi dati e risultati")
        
        # Esporta CSV
        all_reviews_list = st.session_state.data['trustpilot'] + st.session_state.data['google'] + st.session_state.data['tripadvisor']
        if all_reviews_list:
            df = pd.DataFrame(all_reviews_list)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Scarica tutte le recensioni (CSV)", data=csv, file_name="reviews_export.csv", mime="text/csv", use_container_width=True)
        
        # Esporta Report Testuale
        seo_results = st.session_state.data.get('seo_analysis')
        if st.session_state.flags['analysis_done'] and seo_results and 'error' not in seo_results:
            report_text = f"Report Analisi SEO per Boscolo Viaggi - {datetime.now().strftime('%Y-%m-%d')}\n\n"
            report_text += "=== TEMI PRINCIPALI ===\n"
            for theme in seo_results.get('top_themes', []):
                report_text += f"- {theme['theme'].title()}: {theme['description']}\n"
            report_text += "\n=== FAQ SUGGERITE ===\n"
            for faq in seo_results.get('faq_proposals', []):
                report_text += f"D: {faq['question']}\nR: {faq['suggested_answer']}\n\n"
            
            st.download_button("📄 Scarica Report Analisi (TXT)", data=report_text.encode('utf-8'), file_name="seo_report.txt", mime="text/plain", use_container_width=True)
