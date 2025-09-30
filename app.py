#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reviews Analyzer v7.2 - Enterprise Edition
- Fix Retry (allowed_methods = {"GET","POST"})
- Errori DataForSEO leggibili
- Paginazione Trustpilot / TripAdvisor / Google Maps
- Pulsante Analisi azzurro (primary)
- OpenAI con backoff
"""

import streamlit as st
import pandas as pd
import requests, time, json, re, numpy as np, logging
from openai import OpenAI
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============ CONFIG ============
st.set_page_config(page_title="Boscolo Viaggi Reviews", page_icon="✈️", layout="wide")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reviews-app")

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    DFSEO_LOGIN    = st.secrets["DFSEO_LOGIN"]
    DFSEO_PASS     = st.secrets["DFSEO_PASS"]
except KeyError as e:
    st.error(f"⚠️ Manca {e} nei Secrets di Streamlit."); st.stop()

# CSS (bottoni primary azzurri rimangono invariati)
st.markdown("""
<style>
.stApp { background:#0b0b0b; color:#fff; }
.main-header { text-align:center;padding:18px;
  background:linear-gradient(135deg,#005691 0%,#0099FF 25%,#FFD700 75%,#8B5CF6 100%);
  border-radius:16px;margin-bottom:22px;color:#000;font-weight:700;}
section[data-testid="stSidebar"]{background:#111;}
</style>
""", unsafe_allow_html=True)

if "data" not in st.session_state:
    st.session_state.data = {"trustpilot":[], "google":[], "tripadvisor":[], "seo_analysis":None}
if "flags" not in st.session_state:
    st.session_state.flags = {"data_imported":False,"analysis_done":False,"analysis_running":False}

# ============ HTTP session con retry ============
def build_requests_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=0.7,
        status_forcelist=[429,500,502,503,504],
        allowed_methods={"GET","POST"}
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter); s.mount("http://", adapter)
    return s
HTTP = build_requests_session()
API_ROOT="https://api.dataforseo.com/v3"

# ============ Helpers DataForSEO ============
def _extract_dfseo_error(data: dict) -> str:
    try:
        tasks = data.get("tasks") or []
        if tasks and tasks[0].get("error"):
            return f"Task error: {tasks[0]['error']}"
        if tasks and tasks[0].get("status_message"):
            return tasks[0]["status_message"]
        return data.get("status_message","Errore sconosciuto")
    except Exception: return "Errore sconosciuto"

def _dfseo_post(endpoint: str, payload: List[Dict]) -> Dict:
    r = HTTP.post(f"{API_ROOT}/{endpoint}", auth=(DFSEO_LOGIN,DFSEO_PASS), json=payload, timeout=60)
    r.raise_for_status(); return r.json()

def _dfseo_get(url: str) -> Dict:
    r = HTTP.get(url, auth=(DFSEO_LOGIN,DFSEO_PASS), timeout=60)
    r.raise_for_status(); return r.json()

def post_task_and_get_id(endpoint: str, payload: List[Dict]) -> str:
    data=_dfseo_post(endpoint,payload)
    if data.get("tasks_error",0)>0: raise Exception(_extract_dfseo_error(data))
    t=data["tasks"][0]
    if t.get("status_code")!=20000: raise Exception(_extract_dfseo_error(data))
    return t["id"]

def wait_task_and_collect_items(endpoint: str, task_id: str)->Tuple[List[Dict],Optional[str]]:
    url=f"{API_ROOT}/{endpoint}/task_get/{task_id}"
    for _ in range(60):
        time.sleep(10)
        data=_dfseo_get(url); t=data["tasks"][0]; code=t.get("status_code")
        if code==20000:
            items=[]; token=None
            for page in t.get("result",[]):
                items+=page.get("items",[])
                token=page.get("next_page_token") or token
            return items,token
        elif code in (20100,40602) or "queue" in (t.get("status_message","").lower()):
            continue
        else: raise Exception(_extract_dfseo_error(data))
    raise Exception("Timeout task")

# ============ Fetch con paginazione ============
def fetch_trustpilot_reviews(url:str,limit:int)->List[Dict]:
    out=[]; offset=0; per_page=100
    while len(out)<limit:
        batch=min(per_page,limit-len(out))
        pid=post_task_and_get_id("business_data/trustpilot/reviews/task_post",
            [{"url":url,"limit":batch,"offset":offset,"language_code":"it"}])
        items,_=wait_task_and_collect_items("business_data/trustpilot/reviews",pid)
        if not items: break
        out+=items; offset+=len(items)
    return out

def fetch_tripadvisor_reviews(url:str,limit:int)->List[Dict]:
    out=[]; offset=0; per_page=100
    while len(out)<limit:
        batch=min(per_page,limit-len(out))
        pid=post_task_and_get_id("business_data/tripadvisor/reviews/task_post",
            [{"url":url,"limit":batch,"offset":offset,"language_code":"it"}])
        items,_=wait_task_and_collect_items("business_data/tripadvisor/reviews",pid)
        if not items: break
        out+=items; offset+=len(items)
    return out

def fetch_google_reviews(pid:str,limit:int)->List[Dict]:
    out=[]; offset=0; token=None; per_page=100
    while len(out)<limit:
        batch=min(per_page,limit-len(out))
        payload={"place_id":pid,"limit":batch,"language_code":"it"}
        if token: payload["page_token"]=token
        else: payload["offset"]=offset
        tid=post_task_and_get_id("business_data/google/maps/reviews/task_post",[payload])
        items,token=wait_task_and_collect_items("business_data/google/maps/reviews",tid)
        if not items: break
        out+=items; offset+=len(items)
        if not token and len(items)<batch: break
    return out

# ============ AI (OpenAI) ============
def analyze_reviews_for_seo(reviews:List[Dict])->Dict:
    texts=[r.get("review_text") or r.get("text") or "" for r in reviews if (r.get("review_text") or r.get("text"))]
    if len(texts)<3: return {"error":"Recensioni insufficienti"}
    sample="\n---\n".join([t[:300] for t in texts[:12]])
    client=OpenAI(api_key=OPENAI_API_KEY)
    sys="Sei un assistente SEO che produce SOLO JSON valido."
    usr=f"""Analizza queste recensioni su Boscolo Viaggi:
{sample}
Task:
1) 5 temi principali
2) 5 FAQ (domanda+risposta)
3) 3 opportunità SEO (content_type, topic, seo_value)
Rispondi SOLO in JSON con chiavi top_themes, faq_proposals, content_opportunities."""
    def call():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
            temperature=0,max_tokens=800,
            response_format={"type":"json_object"})
    for i in range(5):
        try:
            comp=call(); return json.loads(comp.choices[0].message.content)
        except Exception as e:
            time.sleep(2**i)
    return {"error":"Analisi AI fallita"}

# ============ UI ============
st.markdown("<h1 class='main-header'>✈️ REVIEWS: Boscolo Viaggi</h1>", unsafe_allow_html=True)
tab1,tab2,tab3=st.tabs(["🌍 Import Dati","📊 Analisi","📥 Export"])

with tab1:
    st.subheader("Importa Recensioni")
    tp=st.text_input("URL Trustpilot","https://it.trustpilot.com/review/boscolo.com")
    if st.button("Importa Trustpilot"):
        try: st.session_state.data["trustpilot"]=fetch_trustpilot_reviews(tp,100)
        except Exception as e: st.error(e)
    ta=st.text_input("URL TripAdvisor","https://www.tripadvisor.it/Attraction_Review-g187867-d24108558-Reviews-Boscolo_Viaggi.html")
    if st.button("Importa TripAdvisor"):
        try: st.session_state.data["tripadvisor"]=fetch_tripadvisor_reviews(ta,100)
        except Exception as e: st.error(e)
    gid=st.text_input("Google Place ID","ChIJ-R_d-iV-1BIRsA7DW2s-2GA")
    if st.button("Importa Google"):
        try: st.session_state.data["google"]=fetch_google_reviews(gid,100)
        except Exception as e: st.error(e)

with tab2:
    st.subheader("Analisi SEO & FAQ")
    if st.button("🚀 Avvia Analisi",type="primary"):
        allr=st.session_state.data["trustpilot"]+st.session_state.data["tripadvisor"]+st.session_state.data["google"]
        st.session_state.data["seo_analysis"]=analyze_reviews_for_seo(allr)
    res=st.session_state.data.get("seo_analysis")
    if res:
        if "error" in res: st.error(res["error"])
        else:
            st.json(res)

with tab3:
    if st.session_state.data.get("seo_analysis") and "error" not in st.session_state.data["seo_analysis"]:
        txt="Report SEO "+datetime.now().strftime("%Y-%m-%d")
        st.download_button("📄 Scarica Report",txt.encode(),file_name="seo_report.txt")
