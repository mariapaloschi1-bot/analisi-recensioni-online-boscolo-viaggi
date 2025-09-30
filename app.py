#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reviews Analyzer v7.1 - Enterprise Edition (full refactor)
- Errori DataForSEO mostrati correttamente (no più "Ok.")
- Retry/timeout HTTP, con diagnostica opzionale
- Paginazione Trustpilot / TripAdvisor / Google Maps Reviews
- Pulsante Analisi azzurro (primary) + CSS che non lo sovrascrive
- OpenAI con backoff e controllo token
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
# SIDEBAR / DEBUG
# =========================
with st.sidebar:
    st.header("⚙️ Impostazioni")
    DEBUG_API = st.toggle("Mostra diagnostica API (debug)", value=False, help="Visualizza gli ultimi payload/riposte DataForSEO utili per il troubleshooting.")
    st.caption("Consiglio: tienilo spento in produzione.")

# =========================
# THEME / CSS
# =========================
st.markdown("""
<style>
    .stApp { background-color: #0b0b0b; color: #ffffff; }
    .main-header {
        text-align:center; padding: 18px;
        background: linear-gradient(135deg, #005691 0%, #0099FF 25%, #FFD700 75%, #8B5CF6 100%);
        border-radius: 16px; margin-bottom: 22px; color: #0b0b0b; font-weight: 700;
    }
    section[data-testid="stSidebar"] { background-color: #111; }
    [data-testid="stMetric"] { background: #151515; padding: 12px; border-radius: 10px; border: 1px solid #222; }
    /* NON toccare i bottoni primary (devono restare azzurri) */
    .stButton > button:not(.css-1q8dd3e) {
        background: #1f1f1f; color: #fff; border: 1px solid #2a2a2a;
    }
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
if "last_api_debug" not in st.session_state:
    st.session_state.last_api_debug = {}

# =====================================================================================
# HTTP SESSION CON RETRY
# =====================================================================================
def build_requests_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(['GET', 'P]()_
