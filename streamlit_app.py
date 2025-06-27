import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gtts import gTTS
import tempfile

# CONFIGURATION
st.set_page_config(page_title="Dashboard Crédit Client", layout="wide")
st.title("📊 Dashboard de scoring client – Crédit")
st.markdown("Accessible, lisible, interactif pour les métiers et les personnes en situation de handicap.")

# --- FONCTION AUDIO ---

def lire_decision_audio_web(phrase):
    try:
        tts = gTTS(phrase, lang='fr')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3")
    except Exception as e:
        st.error(f"Erreur lors de la synthèse vocale : {e}")

# --- SIDEBAR ---
st.sidebar.header("🔎 Sélection du client")
client_id = st.sidebar.number_input("ID du client", min_value=0, step=1, help="Identifiant du client à analyser")

variables_comparables = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "DAYS_BIRTH", "CNT_CHILDREN", "CNT_FAM_MEMBERS"
]
var_select = st.sidebar.selectbox("Comparer avec d'autres clients selon :", variables_comparables)

# --- API CALLS ---
url_pred = f"https://projet-opc.onrender.com/prediction/{client_id}"
url_interpret = f"https://projet-opc.onrender.com/interpretabilite/{client_id}"

# --- DONNÉES ---
@st.cache_data
def load_global_data():
    try:
        df = pd.read_csv("data_test.csv")
        df["AGE"] = (-df["DAYS_BIRTH"] / 365).astype(int)
        return df
    except:
        st.error("Impossible de charger les données globales.")
        return pd.DataFrame()

df_global = load_global_data()

# --- PREDICTION ---
st.header("🧠 Résultat du modèle")
proba = None

if st.button("🎯 Obtenir la prédiction pour ce client"):
    try:
        response = requests.get(url_pred, timeout=90)
        response.raise_for_status()
        result = response.json()

        if "error" in result:
            st.error(f"Erreur : {result['error']}")
        else:
            proba = result["probability_default"]
            seuil = 0.5
            label = "✅ Faible risque" if proba < seuil else "❌ Risque élevé"
            couleur = "green" if proba < seuil else "red"

            st.markdown(f"### 🔐 Probabilité de défaut : **{proba:.2%}**")
            st.markdown(
                f"### Interprétation : <span style='color:{couleur}; font-weight:bold'>{label}</span>",
                unsafe_allow_html=True
            )
            st.progress(proba)

            if st.button("🔊 Lire la décision à voix haute"):
                phrase = f"La décision de crédit est : {'accepté' if proba < 0.5 else 'refusé'}. La probabilité de défaut est de {proba:.0%}."
                lire_decision_audio_web(phrase)


    except Exception as e:
        st.error(f"Erreur API : {e}")

# --- INTERPRÉTATION ---
st.header("📌 Interprétation du score (SHAP)")
try:
    response = requests.get(url_interpret, timeout=90)
    response.raise_for_status()
    data = response.json()

    if "error" in data:
        st.error(data["error"])
    else:
        shap_local = data.get("shap_values", {})
        shap_global = data.get("shap_global", {})

        # SHAP LOCAL
        st.subheader("🔍 Importance locale des variables pour ce client")
        shap_series = pd.Series(shap_local).sort_values(key=abs, ascending=False)
        st.bar_chart(shap_series.head(10))
        st.caption("Top 10 des variables expliquant la décision pour ce client.")

        # SHAP GLOBAL
        st.subheader("🌐 Importance globale des variables")
        shap_global_series = pd.Series(shap_global).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=shap_global_series.values, y=shap_global_series.index, palette="viridis", ax=ax)
        ax.set_xlabel("Importance moyenne (|valeurs SHAP|)")
        ax.set_title("Variables les plus influentes dans le modèle")
        ax.set_facecolor("white")
        st.pyplot(fig)
        st.caption("Variables ayant le plus d’impact global dans le modèle.")

except Exception as e:
    st.error(f"Erreur lors de la récupération des SHAP : {e}")

# --- INFOS CLIENT ---
st.header("🧾 Informations client")
if not df_global.empty and client_id in df_global["SK_ID_CURR"].values:
    infos_client = df_global[df_global["SK_ID_CURR"] == client_id].squeeze()
    cols_infos = ["AGE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "CNT_CHILDREN", "CNT_FAM_MEMBERS"]
    st.dataframe(infos_client[cols_infos].to_frame("Valeur").rename_axis("Variable"), use_container_width=True)
else:
    st.warning("Client non trouvé dans la base globale.")

# --- COMPARAISON ---
st.header("📈 Comparaison avec d'autres clients")
if var_select in df_global.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_global[var_select], bins=30, kde=True, color="lightgray", label="Population totale", ax=ax)

    if client_id in df_global["SK_ID_CURR"].values:
        val_client = infos_client[var_select]
        ax.axvline(val_client, color="red", linestyle="--", linewidth=2, label="Client")
        ax.legend()
        ax.set_title(f"Comparaison sur la variable : {var_select}")
        ax.set_facecolor("white")
        st.pyplot(fig)
        st.caption("La ligne rouge représente la position du client par rapport au reste de la population.")
    else:
        st.warning("Impossible de comparer : client absent.")
