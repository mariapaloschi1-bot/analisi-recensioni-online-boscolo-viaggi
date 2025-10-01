#!/usr/bin/env python3
"""
Reviews Analyzer v2.0 ENTERPRISE EDITION
Supports: Trustpilot, Google Reviews, TripAdvisor, Yelp (via Extended Reviews), Reddit
Advanced Analytics: Multi-Dimensional Sentiment, ABSA, Topic Modeling, Customer Journey
Autore: Mari
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
from docx import Document
from openai import OpenAI
from collections import Counter
import os
from urllib.parse import urlparse
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass

# ============================================================================
# CONFIGURAZIONE PAGINA (ANTICIPATA)
# Questo comando deve essere il primo comando Streamlit eseguito.
# ============================================================================
st.set_page_config(
    page_title="Review NLZYR",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GESTIONE CREDENZIALI (SUPER ROBUSTA)
# Questo è il punto più critico che può causare il blocco dell'app.
# ============================================================================
DFSEO_LOGIN = ""
DFSEO_PASS = ""
OPENAI_API_KEY = ""
GEMINI_API_KEY = ""
credentials_loaded = False

try:
    # Questo blocco cerca le credenziali in st.secrets.
    # Se fallisce, l'app mostrerà un errore chiaro invece di una pagina bianca.
    DFSEO_LOGIN = st.secrets["dfseo_login"]
    DFSEO_PASS = st.secrets["dfseo_pass"]
    OPENAI_API_KEY = st.secrets["openai_api_key"]
    GEMINI_API_KEY = st.secrets["gemini_api_key"]
    credentials_loaded = True
    st.sidebar.success("✅ Credenziali caricate.")
except (KeyError, FileNotFoundError):
    # MODIFICA: Questo messaggio è ora molto più visibile e chiaro.
    st.error(
        """
        **ERRORE CRITICO: CREDENZIALI MANCANTI!**

        L'applicazione non può avviarsi perché non trova le credenziali nel file `secrets.toml`.

        **Soluzione:**
        1.  Crea una cartella `.streamlit` nella directory del tuo progetto.
        2.  Al suo interno, crea un file `secrets.toml`.
        3.  Incolla questo testo nel file, sostituendo i valori con le tue chiavi API:

        ```toml
        dfseo_login = "la_tua_email@esempio.com"
        dfseo_pass = "la_tua_password_dataforseo"
        openai_api_key = "sk-..."
        gemini_api_key = "AIzaSy..."
