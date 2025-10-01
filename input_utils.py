import re
import streamlit as st

def extract_google_place_id_from_url(url: str) -> str:
    match = re.search(r'placeid=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    match = re.search(r'!1s([a-zA-Z0-9_-]{20,})!', url)
    if match:
        return match.group(1)
    return None

def validate_extended_input(user_input: str) -> bool:
    if user_input.strip().startswith("http") or "www." in user_input:
        st.error("Per Extended Reviews inserisci SOLO il nome del business, NON un URL.")
        return False
    if len(user_input.strip()) < 3:
        st.error("Il nome del business è troppo corto.")
        return False
    return True

def validate_reddit_input(user_input: str) -> list:
    urls = [u.strip() for u in user_input.splitlines() if u.strip()]
    filtered = []
    for u in urls:
        if "reddit.com" in u.lower():
            st.warning(f"Ignorato URL Reddit: {u}")
        else:
            filtered.append(u)
    return filtered

def handle_google_id_input(google_input: str):
    if google_input.strip().startswith("http"):
        extracted = extract_google_place_id_from_url(google_input)
        if extracted:
            st.info(f"Place ID estratto automaticamente: `{extracted}`")
            return extracted
        else:
            st.error("Impossibile estrarre il Place ID dall’URL. Incolla un URL valido di Google Maps.")
            return None
    return google_input.strip()
