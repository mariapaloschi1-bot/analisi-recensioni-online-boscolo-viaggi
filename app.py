import streamlit as st
import pandas as pd
import random
import json
import time

# --- Configurazione Pagina ---
st.set_page_config(layout="wide", page_title="Analisi Recensioni Boscolo Viaggi")

# --- Setup API Key (Necessario per l'analisi AI) ---
# NOTE: In un'implementazione reale, useresti la libreria Google GenAI SDK (o 'requests') per chiamare l'API.
# Per questa demo, simuleremo la chiamata e l'output per mostrare la struttura dell'applicazione.
try:
    # Tenta di leggere la chiave API dal file secrets.toml
    API_KEY = st.secrets["gemini_api_key"]
except KeyError:
    API_KEY = "CHIAVE_API_NON_CONFIGURATA"
    st.warning("⚠️ **Chiave API Gemini non configurata.** Per l'analisi AI, devi inserire la chiave API nel file `secrets.toml` nella cartella `.streamlit/`.")

# --- Dati di input (Recensioni Simulate) ---
# In un'applicazione reale, dovresti implementare uno scraper per queste URL:
REVIEW_URLS = [
    "https://it.trustpilot.com/review/boscolo.com",
    "https://www.tripadvisor.it/Attraction_Review-g187867-d24108558-Reviews-Boscolo_Viaggi-Padua_Province_of_Padua_Veneto.html",
    "https://www.google.com/search?q=Boscolo+Tours+S.P.A.+Recensioni",
]

# Attributi specifici per Tour Operator
TOUR_OPERATOR_ATTRIBUTES = [
    "Guide ed Esperti Locali",
    "Itinerario e Tappe",
    "Processo di Prenotazione",
    "Assistenza Clienti",
    "Qualità degli Alloggi",
    "Logistica e Trasporti",
    "Rapporto Qualità-Prezzo",
    "Escursioni e Attività"
]

# Simulazione di Recensioni (Sostituisci con recensioni reali dopo lo scraping)
MOCK_REVIEWS = [
    "Il viaggio Boscolo è stato un'esperienza indimenticabile, la nostra guida, Marco, era preparatissima e appassionata. L'itinerario, tuttavia, era un po' troppo affrettato in alcune tappe. [Guide ed Esperti Locali: Positivo, Itinerario e Tappe: Negativo]",
    "Ho trovato il processo di prenotazione online macchinoso e l'assistenza clienti al telefono quasi inesistente. Il prezzo pagato non vale il servizio ricevuto. [Processo di Prenotazione: Negativo, Assistenza Clienti: Negativo, Rapporto Qualità-Prezzo: Negativo]",
    "Ottimi gli hotel selezionati, davvero di lusso, e i trasferimenti sono stati puntuali. L'unica pecca è stata la mancanza di opzioni per le escursioni facoltative. [Qualità degli Alloggi: Positivo, Logistica e Trasporti: Positivo, Escursioni e Attività: Negativo]",
    "Assolutamente fantastico! Tutto organizzato alla perfezione. L'assistenza prima e durante il tour è stata eccellente. Consigliatissimo! [Assistenza Clienti: Positivo, Itinerario e Tappe: Positivo]",
    "Deluso dal rapporto qualità-prezzo. Ci aspettavamo alloggi migliori dato il costo. La guida parlava poco italiano. [Rapporto Qualità-Prezzo: Negativo, Guide ed Esperti Locali: Negativo]",
]

# --- Prompt per il Modello Gemini (Istruzioni Dettagliate per l'AI) ---

def generate_gemini_prompt(brand_name, reviews, attributes):
    """
    Genera il prompt in italiano per il modello Gemini, istruendolo
    sull'analisi delle recensioni e sulla formattazione dell'output JSON.
    """
    reviews_text = "\n---\n".join(reviews)
    attributes_list = ", ".join(attributes)

    system_instruction = (
        "Sei un analista di mercato specializzato nel settore viaggi e tour operator. "
        "Il tuo obiettivo è analizzare un set di recensioni per un brand specifico e "
        "fornire un'analisi strutturata in formato JSON, basata sugli attributi chiave forniti. "
        "Non aggiungere introduzioni o testo libero, solo l'oggetto JSON richiesto. "
    )

    user_query = f"""
    Analizza le seguenti recensioni per il tour operator '{brand_name}'.

    **Recensioni da Analizzare:**
    {reviews_text}

    **Attributi Chiave per l'Analisi (Focus Tour Operator):**
    {attributes_list}

    Dall'analisi delle recensioni, fornisci la risposta nel seguente formato JSON rigoroso:
    {{
      "riassunto_esecutivo": "Un riassunto conciso dei punti di forza e di debolezza del brand. (Max 150 parole)",
      "sentiment_generale": "Un singolo aggettivo in italiano (es: Positivo, Neutro, Negativo).",
      "punti_dolenti": [
        "Elenca 3-4 problemi ricorrenti citati nelle recensioni (es: 'Tariffe non chiare', 'Trasporti in ritardo')."
      ],
      "distribuzione_attributi": [
        {{"attributo": "Nome Attributo 1", "punteggio": "1-5", "commento": "Breve commento sulle performance di questo attributo."}},
        ... (per tutti gli attributi forniti)
      ],
      "raccomandazioni_strategiche": [
        "Fornisci 3 raccomandazioni strategiche e attuabili per migliorare la soddisfazione del cliente in base ai punti dolenti."
      ]
    }}

    Rispondi solo con l'oggetto JSON.
    """

    return system_instruction, user_query

# --- Funzione che simula la chiamata API (JSON Output) ---
# Questa funzione simula la risposta strutturata che ci aspetteremmo dal modello Gemini.

def mock_ai_analysis(brand_name, reviews, attributes):
    """Simula l'analisi AI e l'output JSON."""
    time.sleep(1.5) # Simula un tempo di risposta
    
    # Questo è l'output JSON strutturato che ci aspettiamo dal modello Gemini
    mock_json_response = {
        "riassunto_esecutivo": f"L'analisi delle recensioni per {brand_name} rivela una polarizzazione: i clienti lodano la competenza delle Guide e la qualità degli Alloggi, ma esprimono forte insoddisfazione per il Rapporto Qualità-Prezzo e la complessità del Processo di Prenotazione. L'Assistenza Clienti è un punto critico che necessita di interventi immediati per allinearsi al posizionamento premium del brand.",
        "sentiment_generale": "Neutro-Leggermente Negativo",
        "punti_dolenti": [
            "Mancanza di chiarezza sui costi extra (Rapporto Qualità-Prezzo).",
            "Lentezza e inefficacia dell'Assistenza Clienti telefonica.",
            "Siti web e procedure di Prenotazione macchinose.",
            "Itinerari a volte troppo intensi e frettolosi."
        ],
        "distribuzione_attributi": [
            {"attributo": "Guide ed Esperti Locali", "punteggio": "4", "commento": "Le guide ricevono feedback molto positivi per competenza e passione, sono un punto di forza."},
            {"attributo": "Itinerario e Tappe", "punteggio": "3", "commento": "Le destinazioni sono apprezzate, ma alcuni clienti trovano il ritmo del viaggio eccessivo."},
            {"attributo": "Processo di Prenotazione", "punteggio": "2", "commento": "Il sistema di booking online è descritto come obsoleto e poco intuitivo."},
            {"attributo": "Assistenza Clienti", "punteggio": "1", "commento": "Il punto più debole. Segnalazioni di lunghe attese e risposte insoddisfacenti."},
            {"attributo": "Qualità degli Alloggi", "punteggio": "4", "commento": "Gli hotel e le strutture selezionate sono generalmente di alto livello e molto apprezzate."},
            {"attributo": "Logistica e Trasporti", "punteggio": "3", "commento": "Servizio accettabile, ma con occasionali problemi di puntualità nei trasferimenti."},
            {"attributo": "Rapporto Qualità-Prezzo", "punteggio": "2", "commento": "Molti clienti ritengono che il costo sia troppo elevato rispetto al servizio complessivo offerto."},
            {"attributo": "Escursioni e Attività", "punteggio": "3", "commento": "Le attività incluse sono buone, ma i clienti desiderano maggiore flessibilità o opzioni facoltative."}
        ],
        "raccomandazioni_strategiche": [
            "Digitalizzare e semplificare l'interfaccia di prenotazione online, rendendola mobile-first e trasparente sui costi.",
            "Rinforzare immediatamente il team di Assistenza Clienti, introducendo anche canali di supporto tramite chat in tempo reale.",
            "Rivedere la struttura dei prezzi e/o migliorare l'inclusività dei servizi per giustificare il posizionamento premium (ad esempio, includendo più pasti o escursioni nel pacchetto base)."
        ]
    }
    return mock_json_response

# --- Interfaccia Utente Streamlit ---

st.title("🔍 Analisi del Sentiment delle Recensioni - Boscolo Viaggi")
st.subheader("Report Strategico Basato sugli Attributi Chiave dei Tour Operator")

st.markdown("""
<style>
    .stAlert { border-radius: 10px; }
    .stButton>button {
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    .good-score { color: green; font-weight: bold; }
    .bad-score { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# Sidebar con link e info
st.sidebar.header("Riferimenti")
st.sidebar.info("Brand analizzato: **Boscolo Viaggi**")
st.sidebar.markdown("**URL delle recensioni (Scraping Target):**")
for url in REVIEW_URLS:
    st.sidebar.markdown(f"- [{url.split('/')[2]}...]({url})")

st.sidebar.markdown("---")
st.sidebar.markdown("**Attributi di Focus:**")
st.sidebar.code("\n".join(TOUR_OPERATOR_ATTRIBUTES))

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# Pannello principale
col_exec, col_raw = st.columns([2, 1])

with col_raw:
    st.header("Recensioni Grezze Simulate")
    st.info("⚠️ In una versione operativa, questo pannello mostrerebbe le recensioni scaricate dai siti (Trustpilot, Tripadvisor, Google).")
    review_df = pd.DataFrame({
        "Recensione": MOCK_REVIEWS,
        "Sorgente": [random.choice(["Trustpilot", "Tripadvisor", "Google"])] * len(MOCK_REVIEWS)
    })
    st.dataframe(review_df, height=300, use_container_width=True)
    
    st.markdown("---")
    st.caption("Prompt AI per l'Analisi")
    # Mostra la struttura del prompt all'utente
    sys_prompt, user_prompt = generate_gemini_prompt("Boscolo Viaggi", MOCK_REVIEWS, TOUR_OPERATOR_ATTRIBUTES)
    with st.expander("Vedi Istruzioni AI (Prompt)"):
        st.code(f"System Instruction:\n{sys_prompt}", language="markdown")
        st.code(f"User Query:\n{user_prompt}", language="markdown")


with col_exec:
    st.header("Avvia Analisi con Gemini AI")
    if API_KEY != "CHIAVE_API_NON_CONFIGURATA":
        if st.button("▶️ Esegui Analisi Completa"):
            with st.spinner("Analisi in corso... Il modello Gemini sta processando le recensioni e generando il report strategico."):
                try:
                    # Chiamata alla funzione (simulata)
                    analysis = mock_ai_analysis("Boscolo Viaggi", MOCK_REVIEWS, TOUR_OPERATOR_ATTRIBUTES)
                    st.session_state.analysis_result = analysis
                    st.success("✅ Analisi completata con successo!")
                except Exception as e:
                    st.error(f"Errore durante l'analisi AI: {e}")
    else:
        st.error("Per eseguire l'analisi AI, configura la tua chiave Gemini nel file `secrets.toml`.")


# --- Visualizzazione dei Risultati ---
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    
    st.markdown("---")
    st.header("📊 Risultati dell'Analisi di Sentiment")

    # RIASSUNTO E SENTIMENT
    col_sum, col_sent = st.columns([3, 1])
    with col_sum:
        st.subheader("Riassunto Esecutivo")
        st.info(result['riassunto_esecutivo'])
        
    with col_sent:
        sentiment = result['sentiment_generale']
        # Assegna un colore in base al sentiment simulato
        color = 'green' if 'Positivo' in sentiment else ('#ffc107' if 'Neutro' in sentiment else 'red')
        st.subheader("Sentiment Generale")
        st.markdown(f'<div class="metric-box" style="background-color: {color}; color: white;"><h1>{sentiment}</h1></div>', unsafe_allow_html=True)


    # PUNTI DOLENTI E RACCOMANDAZIONI
    st.markdown("---")
    col_pain, col_rec = st.columns(2)
    
    with col_pain:
        st.subheader("💔 Punti Dolenti Ricorrenti")
        st.markdown("Questi sono i temi più problematici che i clienti riscontrano:")
        st.markdown("".join([f"- **{p}**\n" for p in result['punti_dolenti']]))
        
    with col_rec:
        st.subheader("💡 Raccomandazioni Strategiche (Output AI)")
        st.markdown("Strategie concrete per affrontare le criticità:")
        st.markdown("".join([f"- **{r}**\n" for r in result['raccomandazioni_strategiche']]))

    # ANALISI PER ATTRIBUTO
    st.markdown("---")
    st.subheader("⭐ Distribuzione del Punteggio per Attributo Tour Operator (Scala 1-5)")
    
    attr_data = result['distribuzione_attributi']
    
    for item in attr_data:
        att = item['attributo']
        score = int(item['punteggio'])
        comment = item['commento']
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            score_color = 'green' if score >= 4 else ('orange' if score == 3 else 'red')
            st.markdown(f'<div class="metric-box" style="background-color: #fff; border: 1px solid #ddd; padding: 10px; margin-top: 10px;">'
                        f'<p style="font-size: 16px; margin: 0; font-weight: bold;">{att}</p>'
                        f'<p style="font-size: 32px; margin: 5px 0 0 0; color: {score_color};">{score}/5</p>'
                        f'</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**Insight:** {comment}", help="Commento generato dall'AI sull'andamento dell'attributo.")
