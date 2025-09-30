#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reviews Analyzer v7.0 - Enterprise Edition (refactor + fixes)
Autore: Maria (refactor by ChatGPT)
- FIX TripAdvisor: usa payload con `url` (niente g/d id parsing) + paginazione
- FIX Trustpilot: paginazione con offset
- FIX Google: endpoint corretto `business_data/google/maps/reviews` + page_token
- Migliorie: backoff OpenAI, campionamento controllato, UI/UX, export, robustezza errori
"""

# =========================
# IMPORT
# =========================
import streamlit as st
import pandas as pd
import requests
import time
import json
import re
import numpy as np
import logging
from openai import OpenAI
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Boscolo Viaggi Reviews",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reviews-app")

# =========================
# CREDENZIALI
# =========================
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    DFSEO_LOGIN    = st.secrets["DFSEO_LOGIN"]
    DFSEO_PASS     = st.secrets["DFSEO_PASS"]
except KeyError as e:
    st.error(f"⚠️ Manca una credenziale nei Secrets di Streamlit: {e}. L'app non può funzionare.")
    st.stop()

# =========================
# THEME / CSS
# =========================
st.markdown("""
<style>
    .stApp { background-color: #0b0b0b; color: #FFFFFF; }
    .main-header {
        text-align: center; padding: 18px;
        background: linear-gradient(135deg, #005691 0%, #0099FF 25%, #FFD700 75%, #8B5CF6 100%);
        border-radius: 16px; margin-bottom: 22px; color: #0b0b0b; font-weight: 700;
    }
    section[data-testid="stSidebar"] { background-color: #111; }
    [data-testid="stMetric"] { background: #151515; padding: 12px; border-radius: 10px; border: 1px solid #222; }
    .small-note { color:#aaa; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "data" not in st.session_state:
    st.session_state.data = {
        "trustpilot": [],
        "google": [],
        "tripadvisor": [],
        "seo_analysis": None
    }
if "flags" not in st.session_state:
    st.session_state.flags = {
        "data_imported": False,
        "analysis_done": False,
        "analysis_running": False
    }

# =====================================================================================
# HELPERS HTTP / DATAFORSEO
# =====================================================================================
API_ROOT = "https://api.dataforseo.com/v3"

def _dfseo_post(endpoint: str, payload: List[Dict]) -> Dict:
    url = f"{API_ROOT}/{endpoint}"
    r = requests.post(url, auth=(DFSEO_LOGIN, DFSEO_PASS), json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def _dfseo_get(url: str) -> Dict:
    r = requests.get(url, auth=(DFSEO_LOGIN, DFSEO_PASS), timeout=60)
    r.raise_for_status()
    return r.json()

def post_task_and_get_id(endpoint: str, payload: List[Dict]) -> str:
    data = _dfseo_post(endpoint, payload)
    if data.get("tasks_error", 1) > 0 or not data.get("tasks"):
        msg = (data.get("status_message") or "Errore sconosciuto nella creazione del task")
        raise Exception(f"Errore API (Creazione Task): {msg}")
    task = data["tasks"][0]
    if task.get("status_code") not in (20000, 20100, 40500, 40602):
        raise Exception(f"Errore API (Task): {task.get('status_message')}")
    return task["id"]

def wait_task_and_collect_items(endpoint: str, task_id: str) -> Tuple[List[Dict], Optional[str]]:
    """
    Attende il completamento del task e ritorna (items, next_page_token_se_presente)
    Alcuni endpoint (Google) usano next_page_token per paginare.
    """
    result_url = f"{API_ROOT}/{endpoint}/task_get/{task_id}"
    for attempt in range(60):  # ~10 min se sleep=10
        time.sleep(10)
        data = _dfseo_get(result_url)
        if not data.get("tasks"):
            continue
        task = data["tasks"][0]
        code = task.get("status_code")
        msg  = (task.get("status_message") or "").lower()
        if code == 20000:
            items: List[Dict] = []
            next_token: Optional[str] = None
            results = task.get("result") or []
            for page in results:
                if page.get("items"):
                    items.extend(page["items"])
                # next_page_token può essere a livello pagina
                if page.get("next_page_token"):
                    next_token = page["next_page_token"]
            return items, next_token
        elif code in (20100, 40602) or "queue" in msg or "handed" in msg:
            continue
        else:
            raise Exception(f"Stato task non valido: {code} - {task.get('status_message')}")
    raise Exception("Timeout: il task ha impiegato troppo tempo.")

# =====================================================================================
# FETCH FUNZIONI (CON PAGINAZIONE)
# =====================================================================================
def fetch_trustpilot_reviews(tp_url: str, limit: int) -> List[Dict]:
    collected: List[Dict] = []
    offset = 0
    # Alcuni account hanno per_page=20; proviamo 100 e scaliamo con quello che torna
    per_page = 100
    while len(collected) < limit:
        batch_limit = min(per_page, limit - len(collected))
        payload = [{
            "url": tp_url,
            "limit": int(batch_limit),
            "offset": int(offset),
            "language_code": "it"
        }]
        task_id = post_task_and_get_id("business_data/trustpilot/reviews/task_post", payload)
        items, _ = wait_task_and_collect_items("business_data/trustpilot/reviews", task_id)
        if not items:
            break
        collected.extend(items)
        offset += len(items)
        if len(items) < batch_limit:
            break
    return collected

def fetch_tripadvisor_reviews(ta_url: str, limit: int) -> List[Dict]:
    """
    TripAdvisor: usare payload con `url` (NON parsing -g / -d).
    Alcuni account hanno limit max per task ~100: loopiamo.
    """
    collected: List[Dict] = []
    # TripAdvisor non sempre espone token; usiamo offset quando disponibile
    offset = 0
    per_page = 100
    while len(collected) < limit:
        batch_limit = min(per_page, limit - len(collected))
        payload = [{
            "url": ta_url,
            "limit": int(batch_limit),
            "offset": int(offset),
            "sort_by": "date_desc",
            "language_code": "it"
        }]
        task_id = post_task_and_get_id("business_data/tripadvisor/reviews/task_post", payload)
        items, _ = wait_task_and_collect_items("business_data/tripadvisor/reviews", task_id)
        if not items:
            break
        collected.extend(items)
        offset += len(items)
        if len(items) < batch_limit:
            break
    return collected

def fetch_google_reviews(place_id: str, limit: int) -> List[Dict]:
    """
    Google Maps Reviews via DataForSEO:
    endpoint: business_data/google/maps/reviews
    Paginazione con next_page_token quando presente, altrimenti offset.
    """
    collected: List[Dict] = []
    next_token: Optional[str] = None
    offset = 0
    per_page = 100

    while len(collected) < limit:
        batch_limit = min(per_page, limit - len(collected))
        payload = [{
            "place_id": place_id,
            "limit": int(batch_limit),
            "language_code": "it",
            **({"page_token": next_token} if next_token else {}),
            **({} if next_token else {"offset": int(offset)}),
        }]
        task_id = post_task_and_get_id("business_data/google/maps/reviews/task_post", payload)
        items, next_token = wait_task_and_collect_items("business_data/google/maps/reviews", task_id)
        if not items:
            break
        collected.extend(items)
        # Se c'è token, continuiamo con token; se non c'è, andiamo di offset
        if next_token:
            continue
        offset += len(items)
        if len(items) < batch_limit:
            break
    return collected

# =====================================================================================
# AI: ANALISI SEO (OpenAI) CON BACKOFF
# =====================================================================================
def _retry_backoff(func, max_tries=5, base_delay=1.5, exc_types=(Exception,), on_error_msg=None):
    tries = 0
    while True:
        try:
            return func()
        except exc_types as e:
            tries += 1
            if tries >= max_tries:
                if on_error_msg:
                    logger.error(on_error_msg + f" Dettagli: {e}")
                raise
            sleep_s = base_delay ** tries + np.random.rand() * 0.3
            time.sleep(sleep_s)

def analyze_reviews_for_seo(reviews: List[Dict]) -> Dict:
    with st.spinner("Esecuzione analisi SEO e generazione FAQ con AI..."):
        # Normalizza i campi testuali più comuni
        texts = []
        for r in reviews:
            t = r.get("review_text") or r.get("text") or r.get("content") or r.get("body") or ""
            t = str(t).strip()
            if t:
                texts.append(t)
        if len(texts) < 3:
            return {"error": "Dati insufficienti per un’analisi credibile (minimo 3 recensioni non vuote)."}

        # Campionamento controllato
        SAMPLE_N = min(12, len(texts))
        sample_reviews_text = "\n---\n".join([t[:300] for t in texts[:SAMPLE_N]])

        client = OpenAI(api_key=OPENAI_API_KEY)

        sys_msg = "Sei un assistente SEO che fornisce esclusivamente output JSON valido (UTF-8)."
        user_prompt = f"""
Analizza queste recensioni reali per 'Boscolo Viaggi'.

RECENSIONI (ESTRATTI):
{sample_reviews_text}

TASK:
1) Estrai i 5 temi più importanti (con breve descrizione).
2) Genera 5 proposte di FAQ con risposta suggerita (Q/A sintetica).
3) Indica 3 opportunità di contenuto SEO con: content_type, topic, seo_value (basso/medio/alto).

RISPONDI SOLO CON JSON con chiavi:
- "top_themes": [{{"theme": str, "description": str}} x5]
- "faq_proposals": [{{"question": str, "suggested_answer": str}} x5]
- "content_opportunities": [{{"content_type": str, "topic": str, "seo_value": str}} x3]
""".strip()

        def _call_openai():
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=800,
                response_format={"type": "json_object"},
            )

        try:
            completion = _retry_backoff(
                _call_openai,
                max_tries=5,
                base_delay=1.7,
                exc_types=(Exception,),
                on_error_msg="Errore durante la chiamata OpenAI (dopo vari tentativi)."
            )
            content = completion.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            # Messaggio pulito all'utente; dettagli in log
            logger.exception("OpenAI failure")
            return {"error": "Analisi AI fallita (limiti o errore temporaneo). Riprova tra poco."}

# =====================================================================================
# UTIL: PROGRESS WRAPPER (sincrono, testato con Streamlit)
# =====================================================================================
def safe_api_call_with_progress(label: str, api_function, *args, **kwargs):
    status = st.empty()
    bar = st.progress(0, text=f"{label} — avvio…")
    result = None
    error = None
    try:
        bar.progress(10, text=f"{label} — creazione task…")
        result = api_function(*args, **kwargs)
        bar.progress(100, text=f"{label} — completato")
    except Exception as e:
        error = e
    finally:
        time.sleep(0.3)
        bar.empty()
        status.empty()
    if error:
        raise error
    return result

# =====================================================================================
# UI
# =====================================================================================
st.markdown("<h1 class='main-header'>✈️ REVIEWS: Boscolo Viaggi — v7</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🌍 Import Dati", "📊 Dashboard Analisi", "📥 Export"])

# -------------------------
# TAB 1: Import Dati
# -------------------------
with tab1:
    st.markdown("### 🌍 Importa Dati Reali dalle Piattaforme")

    c1, c2 = st.columns(2)

    with c1.expander("🌟 Trustpilot", expanded=True):
        tp_url = st.text_input(
            "URL Trustpilot",
            "https://it.trustpilot.com/review/boscolo.com",
            key="tp_url_input"
        )
        tp_limit = st.slider("Max Recensioni TP", 20, 1000, 120, step=20, key="tp_slider")
        if st.button("Importa da Trustpilot", use_container_width=True):
            try:
                reviews = safe_api_call_with_progress(
                    "Import Trustpilot",
                    fetch_trustpilot_reviews,
                    tp_url, tp_limit
                )
                st.session_state.data["trustpilot"] = reviews or []
                st.session_state.flags["data_imported"] = True
                st.success(f"✅ Importate {len(reviews)} recensioni Trustpilot.")
            except Exception as e:
                st.error(f"Errore Trustpilot: {e}")

    with c2.expander("✈️ TripAdvisor", expanded=True):
        ta_url = st.text_input(
            "URL TripAdvisor",
            "https://www.tripadvisor.it/Attraction_Review-g187867-d24108558-Reviews-Boscolo_Viaggi-Padua_Province_of_Padua_Veneto.html",
            key="ta_url_input"
        )
        ta_limit = st.slider("Max Recensioni TA", 20, 1000, 120, step=20, key="ta_slider")
        if st.button("Importa da TripAdvisor", use_container_width=True):
            try:
                reviews = safe_api_call_with_progress(
                    "Import TripAdvisor",
                    fetch_tripadvisor_reviews,
                    ta_url, ta_limit
                )
                st.session_state.data["tripadvisor"] = reviews or []
                st.session_state.flags["data_imported"] = True
                st.success(f"✅ Importate {len(reviews)} recensioni TripAdvisor.")
            except Exception as e:
                st.error(f"Errore TripAdvisor: {e}")

    with st.expander("📍 Google Reviews"):
        g_place_id = st.text_input(
            "Google Place ID",
            "ChIJ-R_d-iV-1BIRsA7DW2s-2GA",  # Boscolo Tours S.P.A.
            key="g_id_input",
            help="Place ID Google Maps del business."
        )
        g_limit = st.slider("Max Recensioni Google", 20, 1000, 120, step=20, key="g_slider")
        if st.button("Importa da Google", use_container_width=True):
            try:
                reviews = safe_api_call_with_progress(
                    "Import Google",
                    fetch_google_reviews,
                    g_place_id, g_limit
                )
                st.session_state.data["google"] = reviews or []
                st.session_state.flags["data_imported"] = True
                st.success(f"✅ Importate {len(reviews)} recensioni Google.")
            except Exception as e:
                st.error(f"Errore Google: {e}")

    st.markdown("---")
    st.subheader("Riepilogo Dati Importati")
    counts = {
        "Trustpilot": len(st.session_state.data["trustpilot"]),
        "Google": len(st.session_state.data["google"]),
        "TripAdvisor": len(st.session_state.data["tripadvisor"]),
    }
    total_items = sum(counts.values())
    if total_items > 0:
        active = [p for p, c in counts.items() if c > 0]
        cols = st.columns(max(1, len(active)))
        for i, platform in enumerate(active):
            cols[i].metric(label=f"📝 {platform}", value=counts[platform])
        st.caption(f"Totale recensioni importate: **{total_items}**")

# -------------------------
# TAB 2: Analisi
# -------------------------
with tab2:
    st.header("📊 Dashboard Analisi")
    if not st.session_state.flags["data_imported"]:
        st.info("⬅️ Importa prima dei dati dal tab 'Import Dati'.")
    else:
        # Evita doppio run su rerun
        if not st.session_state.flags["analysis_done"] and not st.session_state.flags["analysis_running"]:
            if st.button("🚀 Esegui Analisi SEO e Generazione FAQ (AI)", type="primary", use_container_width=True):
                st.session_state.flags["analysis_running"] = True
                all_reviews = (
                    st.session_state.data["trustpilot"]
                    + st.session_state.data["google"]
                    + st.session_state.data["tripadvisor"]
                )
                if len(all_reviews) >= 3:
                    res = analyze_reviews_for_seo(all_reviews)
                    st.session_state.data["seo_analysis"] = res
                    st.session_state.flags["analysis_done"] = ("error" not in (res or {}))
                else:
                    st.session_state.data["seo_analysis"] = {"error": "Non ci sono abbastanza recensioni per l’analisi."}
                st.session_state.flags["analysis_running"] = False
                st.experimental_rerun()

        st.markdown("---")

        # Mostra risultati
        seo_results = st.session_state.data.get("seo_analysis")
        if seo_results:
            if "error" in seo_results:
                st.error(f"Errore durante l'analisi SEO: {seo_results['error']}")
            else:
                st.subheader("📈 Risultati Analisi SEO & Contenuti")

                with st.expander("🔥 Temi Principali Estratti", expanded=True):
                    for theme in seo_results.get("top_themes", []):
                        st.markdown(f"- **{theme.get('theme','').title()}** — *{theme.get('description','')}*")

                with st.expander("❓ Proposte di FAQ (AI)", expanded=True):
                    for i, faq in enumerate(seo_results.get("faq_proposals", []), 1):
                        st.markdown(f"**Q{i}. {faq.get('question','')}**")
                        st.info(faq.get("suggested_answer", ""))
                        st.markdown("---")

                with st.expander("💡 Opportunità di Contenuto SEO"):
                    for idea in seo_results.get("content_opportunities", []):
                        ct = idea.get("content_type", "")
                        topic = idea.get("topic", "")
                        val = idea.get("seo_value", "")
                        st.success(f"**{ct}** sul tema **{topic}** · Valore SEO: **{val}**")

# -------------------------
# TAB 3: Export
# -------------------------
with tab3:
    st.header("📥 Export")
    if not st.session_state.flags["analysis_done"]:
        st.info("Esegui prima un'analisi per abilitare l'export.")
    else:
        st.subheader("Esporta i dati e risultati")

        # Esporta CSV recensioni
        all_reviews = (
            st.session_state.data["trustpilot"]
            + st.session_state.data["google"]
            + st.session_state.data["tripadvisor"]
        )
        if all_reviews:
            df = pd.DataFrame(all_reviews)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Scarica tutte le recensioni (CSV)",
                data=csv,
                file_name="reviews_export.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Esporta Report Testuale SEO
        seo_results = st.session_state.data.get("seo_analysis")
        if seo_results and "error" not in seo_results:
            report_lines = []
            report_lines.append(f"Report Analisi SEO per Boscolo Viaggi - {datetime.now().strftime('%Y-%m-%d')}")
            report_lines.append("")
            report_lines.append("=== TEMI PRINCIPALI ===")
            for t in seo_results.get("top_themes", []):
                report_lines.append(f"- {t.get('theme','').title()}: {t.get('description','')}")
            report_lines.append("")
            report_lines.append("=== FAQ SUGGERITE ===")
            for faq in seo_results.get("faq_proposals", []):
                report_lines.append(f"D: {faq.get('question','')}")
                report_lines.append(f"R: {faq.get('suggested_answer','')}")
                report_lines.append("")
            report_lines.append("=== OPPORTUNITÀ DI CONTENUTO SEO ===")
            for idea in seo_results.get("content_opportunities", []):
                report_lines.append(f"- {idea.get('content_type','')} · {idea.get('topic','')} · Valore: {idea.get('seo_value','')}")
            report_text = "\n".join(report_lines)

            st.download_button(
                "📄 Scarica Report Analisi (TXT)",
                data=report_text.encode("utf-8"),
                file_name="seo_report.txt",
                mime="text/plain",
                use_container_width=True
            )

# =========================
# FOOTER
# =========================
st.caption("© Boscolo Viaggi Reviews · v7 · DataForSEO + OpenAI · Made with ❤️")
st.caption("Se incontri limiti di quota, riprova tra qualche minuto o riduci il campione.")
