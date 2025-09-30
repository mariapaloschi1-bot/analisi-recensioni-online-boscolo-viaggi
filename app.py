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
from typing import Dict, List, Any

# --- CONFIGURAZIONE PAGINA E CREDENZIALI ---
st.set_page_config(page_title="Boscolo Viaggi Reviews", page_icon="✈️", layout="wide")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Le credenziali sono caricate da st.secrets.
try:
    HF_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
    DFSEO_LOGIN = st.secrets["DFSEO_LOGIN"]
    DFSEO_PASS = st.secrets["DFSEO_PASS"]
except KeyError as e:
    st.error(f"⚠️ Manca una credenziale nei Secrets: {e}. Assicurati di avere il file `secrets.toml` configurato.")
    st.stop()

# CSS e Session State
st.markdown("""
<style>
.stApp{background-color:#000;color:#FFF}
.main-header{text-align:center;padding:20px;background:linear-gradient(135deg,#005691 0%,#0099FF 25%,#FFD700 75%,#8B5CF6 100%);border-radius:20px;margin-bottom:30px;color:black;}
.stButton>button{background-color:#0099FF;color:#FFF;border:none}
section[data-testid=stSidebar]{background-color:#1A1A1A}
[data-testid=stMetric]{background-color:#1a1a1a;padding:15px;border-radius:10px}
</style>
""", unsafe_allow_html=True)
if 'data' not in st.session_state:
    st.session_state.data = {'google': [], 'tripadvisor': [], 'seo_analysis': None, 'cleaned_reviews': []}
if 'flags' not in st.session_state:
    st.session_state.flags = {'data_imported': False, 'analysis_done': False}

# ============================================================================
# FUNZIONI API E ANALISI
# ============================================================================

def api_live_call(api_name: str, endpoint: str, payload: List[Dict]) -> List[Dict]:
    """
    Funzione generica per chiamare gli endpoint live di DataForSEO.
    Aggiunge logging di debug per gli errori 404.
    """
    url = f"https://api.dataforseo.com/v3/{endpoint}"
    logger.info(f"Tentativo chiamata API a: {url}")
    
    with st.spinner(f"Connessione a DataForSEO per {api_name}... (può richiedere fino a 2 minuti)"):
        time.sleep(1) 
        
        try:
            response = requests.post(url, auth=(DFSEO_LOGIN, DFSEO_PASS), json=payload, timeout=120)
            response.raise_for_status() # Solleva eccezione per 4xx/5xx
        
        except requests.exceptions.HTTPError as e:
            # Cattura l'errore HTTP (es. 404) e logga il contenuto per debug
            error_message = f"Errore nell'importazione {api_name}: {e}"
            try:
                # Tenta di leggere il corpo della risposta per messaggi di errore specifici di DataForSEO
                error_data = response.json()
                error_message += f"\nContenuto risposta: {error_data}"
            except (json.JSONDecodeError, UnboundLocalError):
                error_message += f"\nRisposta non JSON o vuota per l'errore HTTP."

            logger.error(error_message)
            # Rilancia l'eccezione con il messaggio originale
            raise Exception(error_message)
        
        except Exception as e:
            # Cattura altri errori (es. timeout, connessione)
            raise Exception(f"Errore generico API {api_name}: {e}")
        
        # Gestione risposta JSON DataForSEO
        data = response.json()
        
        if data.get("tasks_error", 0) > 0:
            raise Exception(f"Errore API DataForSEO (tasks_error): {data.get('tasks', [{}])[0].get('status_message', 'Errore sconosciuto')}")
        
        task = data["tasks"][0]
        if task['status_code'] != 20000:
            raise Exception(f"Errore API (status_code {task['status_code']}): {task.get('status_message', 'Errore sconosciuto')}")
            
        items = []
        if task.get("result"):
            for page in task["result"]:
                source = "Google" if "google" in endpoint else "TripAdvisor"
                if page and page.get("items"): 
                    for item in page["items"]:
                        item['source'] = source
                        items.append(item)
        return items

def fetch_google_reviews(place_id: str, limit: int) -> List[Dict]:
    """Recupera recensioni da Google Business Profile (tentativo con l'originale business_data)."""
    payload = [{"place_id": place_id, "limit": limit, "language_code": "it"}]
    # Ritorno all'endpoint originale business_data
    return api_live_call("Google", "business_data/google/reviews/live", payload)

def fetch_tripadvisor_reviews(ta_url: str, limit: int) -> List[Dict]:
    """Recupera recensioni da TripAdvisor."""
    clean_url = ta_url.split('?')[0]
    payload = [{"url": clean_url, "limit": limit}]
    # Endpoint TripAdvisor
    return api_live_call("TripAdvisor", "business_data/tripadvisor/reviews/live", payload)

def analyze_reviews_with_huggingface(reviews: List[Dict]) -> Dict[str, Any]:
    """
    Analizza un batch di recensioni usando Mixtral 8x7B.
    """
    with st.spinner("Esecuzione analisi con modello open-source (potrebbe richiedere tempo)..."):
        all_texts = [r.get('review_text', '') for r in reviews if r.get('review_text')]
        if len(all_texts) < 3: 
            return {'error': 'Dati insufficienti per l\'analisi (meno di 3 recensioni).'}
            
        client = InferenceClient(token=HF_TOKEN)
        model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1" 
            
        sample_reviews_text = "\n---\n".join([r[:500] for r in all_texts[:50]]) 
            
        prompt = f"""
        <s>[INST] Sei un esperto SEO. Analizza le seguenti recensioni per 'Boscolo Viaggi'.
        
        RECENSIONI:
        {sample_reviews_text}
        
        TASK:
        1. Estrai i 3 temi più importanti (es. "Assistenza Clienti", "Qualità Alloggi", "Prezzo").
        2. Genera 3 proposte di FAQ basate sui temi per un sito web. Includi sia la domanda che una breve risposta.
        3. Identifica 2 opportunità di contenuto SEO (es. articoli di blog, guide) per intercettare la domanda implicita nelle recensioni.
        
        Rispondi ESCLUSIVAMENTE in formato JSON valido con le chiavi "top_themes" (lista di stringhe), "faq_proposals" (lista di oggetti con "question" e "answer"), "content_opportunities" (lista di stringhe). Non aggiungere altro testo, introduzioni o spiegazioni. [/INST]
        """
            
        try:
            response = client.text_generation(prompt, model=model_id, max_new_tokens=2048, temperature=0.1)
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if not json_match:
                clean_response = response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response.strip('```json').strip('```').strip()
                    return json.loads(clean_response)
                
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

# --- TAB 1: Import Dati ---
with tab1:
    st.markdown("### 🌍 Importa Dati Reali dalle Piattaforme")
    
    col1, col2 = st.columns(2)
    
    # === Google Reviews ===
    with col1:
        st.subheader("Google Business Profile (Place ID)")
        google_place_id = st.text_input("Place ID di Google (es. ChIJX...)", value="ChIJXb7yX2vFhkcRM_p9lFq44rQ") 
        google_limit = st.slider("Numero max di recensioni Google da importare", min_value=1, max_value=500, value=100)
        
        if st.button("📥 Importa Recensioni Google", use_container_width=True, key="btn_google"):
            if google_place_id:
                try:
                    google_reviews = fetch_google_reviews(google_place_id, google_limit)
                    st.session_state.data['google'] = google_reviews
                    st.session_state.flags['data_imported'] = True
                    st.success(f"✅ Importate {len(google_reviews)} recensioni da Google.")
                except Exception as e:
                    st.error(f"Errore nell'importazione Google: {e}")
            else:
                st.warning("Inserisci un Place ID valido.")

    # === TripAdvisor Reviews ===
    with col2:
        st.subheader("TripAdvisor (URL)")
        ta_url = st.text_input("URL della pagina di TripAdvisor", value="[https://www.tripadvisor.it/Attraction_Review-g187895-d602324-Reviews-Boscolo_Tours-Florence_Tuscany.html](https://www.tripadvisor.it/Attraction_Review-g187895-d602324-Reviews-Boscolo_Tours-Florence_Tuscany.html)") 
        ta_limit = st.slider("Numero max di recensioni TA da importare", min_value=1, max_value=500, value=50)

        if st.button("📥 Importa Recensioni TripAdvisor", use_container_width=True, key="btn_ta"):
            if ta_url and "tripadvisor" in ta_url:
                try:
                    ta_reviews = fetch_tripadvisor_reviews(ta_url, ta_limit)
                    st.session_state.data['tripadvisor'] = ta_reviews
                    st.session_state.flags['data_imported'] = True
                    st.success(f"✅ Importate {len(ta_reviews)} recensioni da TripAdvisor.")
                except Exception as e:
                    st.error(f"Errore nell'importazione TripAdvisor: {e}")
            else:
                st.warning("Inserisci un URL di TripAdvisor valido.")

    st.markdown("---")
    st.subheader("Riepilogo Dati Importati")
    
    total_google = len(st.session_state.data['google'])
    total_ta = len(st.session_state.data['tripadvisor'])
    total_all = total_google + total_ta
    
    col_g, col_t, col_tot, col_status = st.columns(4)
    
    col_g.metric("Recensioni Google", total_google)
    col_t.metric("Recensioni TripAdvisor", total_ta)
    col_tot.metric("Totale Recensioni", total_all)
    
    if total_all > 0:
        col_status.metric("Stato Importazione", "Pronto per l'Analisi", delta="AI Ready", delta_color="normal")
        st.session_state.flags['data_imported'] = True
        
        # Prepara la lista di recensioni pulite per l'analisi
        all_reviews_list = st.session_state.data['google'] + st.session_state.data['tripadvisor']
        all_reviews_df = pd.DataFrame(all_reviews_list)
        st.session_state.data['cleaned_reviews'] = all_reviews_df[
            all_reviews_df.get('review_text', '').astype(str).str.strip() != ''
        ].to_dict('records')
        
    else:
        col_status.metric("Stato Importazione", "In attesa di dati", delta="0 Recensioni", delta_color="inverse")
        st.session_state.flags['data_imported'] = False


    if total_all > 0:
        with st.expander("Anteprima Recensioni Importate"):
            all_reviews_df = pd.DataFrame(st.session_state.data['google'] + st.session_state.data['tripadvisor'])
            
            cols_map = {
                'review_text': 'Recensione', 
                'rating': 'Rating', 
                'author_name': 'Autore',
                'source': 'Fonte'
            }
            
            cols_to_show = {k: v for k, v in cols_map.items() if k in all_reviews_df.columns}
            
            if cols_to_show:
                st.dataframe(
                    all_reviews_df.rename(columns=cols_to_show).head(20).fillna('N/A')[list(cols_to_show.values())], 
                    use_container_width=True
                )
            else:
                 st.warning("Il DataFrame importato non contiene le colonne previste.")

# --- TAB 2: Dashboard Analisi ---
with tab2:
    st.header("📊 Dashboard Analisi")
    
    reviews_for_analysis = st.session_state.data.get('cleaned_reviews', [])
    
    if not st.session_state.flags.get('data_imported', False) or len(reviews_for_analysis) == 0:
        st.info("⬅️ Importa dati con testo per eseguire un'analisi.")
    else:
        
        if len(reviews_for_analysis) < 3:
            st.warning(f"Sono state trovate solo {len(reviews_for_analysis)} recensioni con testo. L'analisi AI richiede almeno 3 recensioni.")
            st.session_state.data['seo_analysis'] = None
            st.session_state.flags['analysis_done'] = False
        
        elif st.session_state.data['seo_analysis'] is None:
            if st.button(f"🚀 Esegui Analisi con AI Open-Source ({len(reviews_for_analysis)} Recensioni)", type="primary", use_container_width=True):
                try:
                    analysis_result = analyze_reviews_with_huggingface(reviews_for_analysis)
                    
                    if analysis_result.get('error'):
                         st.error(f"Errore di analisi: {analysis_result['error']}")
                    else:
                        st.session_state.data['seo_analysis'] = analysis_result
                        st.session_state.flags['analysis_done'] = True
                        st.success("Analisi completata!"); st.balloons(); time.sleep(1); st.rerun()
                except Exception as e:
                    st.error(f"Errore durante l'analisi: {e}")
        
        
        # === Visualizzazione Risultati ===
        if st.session_state.flags.get('analysis_done', False) and st.session_state.data['seo_analysis']:
            analysis = st.session_state.data['seo_analysis']
            
            st.markdown("## Risultati dell'Analisi AI per la SEO")
            st.markdown("---")
            
            # 1. Temi Principali
            st.subheader("🎯 3 Temi Principali emersi dalle Recensioni")
            if 'top_themes' in analysis:
                col_t1, col_t2, col_t3 = st.columns(3)
                cols = [col_t1, col_t2, col_t3]
                for i, theme in enumerate(analysis['top_themes'][:3]):
                    cols[i].markdown(f"**Tema #{i+1}**")
                    cols[i].success(theme)
            
            st.markdown("---")
            
            # 2. Proposte FAQ
            st.subheader("❓ Proposte FAQ (Knowledge Base/Sito)")
            if 'faq_proposals' in analysis:
                for i, faq in enumerate(analysis['faq_proposals']):
                    q = faq.get('question', f"Domanda {i+1} mancante")
                    a = faq.get('answer', "Risposta mancante")
                    with st.expander(f"**{i+1}. {q}**"):
                        st.markdown(a)
                        
            st.markdown("---")

            # 3. Opportunità SEO
            st.subheader("💡 2 Opportunità di Contenuto SEO (Idee Blog/Landing Page)")
            if 'content_opportunities' in analysis:
                for i, opportunity in enumerate(analysis['content_opportunities'][:2]):
                    st.info(f"**Opportunità SEO #{i+1}:** {opportunity}")
                    
            st.markdown("---")

            with st.expander("Visualizza Output JSON Completo (Tecnico)"):
                st.json(analysis)

# --- TAB 3: Export ---
with tab3:
    st.header("📥 Export Dati e Analisi")
    
    google_data = st.session_state.data['google']
    ta_data = st.session_state.data['tripadvisor']
    analysis_data = st.session_state.data['seo_analysis']
    
    st.subheader("Export Recensioni Grezze (CSV)")
    
    col_e1, col_e2 = st.columns(2)
    
    if google_data:
        google_df = pd.DataFrame(google_data).to_csv(index=False).encode('utf-8')
        col_e1.download_button(
            label=f"Scarica {len(google_data)} Recensioni Google (CSV)",
            data=google_df,
            file_name="boscolo_viaggi_google_reviews.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        col_e1.info("Nessun dato Google da esportare.")

    if ta_data:
        ta_df = pd.DataFrame(ta_data).to_csv(index=False).encode('utf-8')
        col_e2.download_button(
            label=f"Scarica {len(ta_data)} Recensioni TripAdvisor (CSV)",
            data=ta_df,
            file_name="boscolo_viaggi_tripadvisor_reviews.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        col_e2.info("Nessun dato TripAdvisor da esportare.")
        
    st.markdown("---")
    
    st.subheader("Export Risultati Analisi AI (JSON)")
    if analysis_data:
        json_string = json.dumps(analysis_data, indent=4)
        st.download_button(
            label="Scarica Analisi SEO AI (JSON)",
            data=json_string,
            file_name="boscolo_viaggi_seo_analysis.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.warning("Esegui prima l'analisi nella 'Dashboard Analisi' per poter esportare i risultati.")
