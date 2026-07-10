# -*- coding: utf-8 -*-
"""
Dashboard Streamlit — Analytics sondaggio "Summer of AI"
Avvio: python -m streamlit run app.py
Dipendenze:  pip install streamlit pandas plotly wordcloud matplotlib
"""

import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------------------------------------------------------- CONFIG ---

st.set_page_config(page_title="Summer of AI — Analytics", page_icon="☀️", layout="wide")

DEFAULT_CSV = "2026_06_Summer_of_AI_-_Info_iscritti_Submissions_2026-07-10.csv"

COL_LIVELLO = "A che livello sei oggi con l'AI?"
COL_MOTIVO = "Perché vuoi imparare a usare meglio l’AI?"
COL_AMBITO = "In quale ambito ti piacerebbe applicare l'AI?"
COL_TOOL = "(opzionale) C'è qualche tool o sua applicazione che ti piacerebbe vedere?"
COL_ETA = "Quanti anni hai?"
COL_OCCUP = "Qual è la tua occupazione?"
COL_AREA = "In quale area lavori?"

EXCLUDE_COLS = {"Submission ID", "Respondent ID", "Submitted at", "e"}

# Opzioni note delle domande multi-risposta (servono per splittare correttamente,
# dato che le opzioni "Ambito" contengono virgole al loro interno)
OPZIONI_MOTIVO = [
    "Per essere più produttivo",
    "Per essere più competitivo sul mercato",
    "Per reinventarmi professionalmente",
    "Per pura curiosità",
]
OPZIONI_AMBITO = [
    "Tech: per scrivere codice, creare software o gestire database",
    "Business: per gestire email, analizzare dati o scrivere report",
    "Creatività: per generare immagini e video o scrivere testi",
    "Studio: per preparare esami, fare ricerche o imparare nuove lingue",
    "Hobby: per pianificare viaggi, gestire le finanze personali o progetti personali",
]

ORDINE_LIVELLO = ["Zero", "Base", "Intermedio", "Avanzato", "Master"]

PALETTE = px.colors.qualitative.Set2


# ----------------------------------------------------------------- UTILS ---

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    return df


def estrai_opzioni(risposta: str, opzioni_note: list[str]) -> list[str]:
    """Estrae le opzioni note contenute in una risposta multi-select.
    Eventuale testo residuo non riconosciuto viene ignorato (rumore/typo)."""
    if not isinstance(risposta, str):
        return []
    trovate = [opt for opt in opzioni_note if opt in risposta]
    return trovate


def barre_con_percentuali(counts: pd.Series, titolo: str, base_pct: int,
                          nota_pct: str, orientamento: str = "h",
                          ordine: list[str] | None = None):
    """Istogramma con etichetta 'conteggio (xx.x%)' su ogni barra."""
    data = counts.reset_index()
    data.columns = ["Risposta", "Conteggio"]
    data["Percentuale"] = data["Conteggio"] / base_pct * 100
    data["label"] = data.apply(lambda r: f"{r['Conteggio']} ({r['Percentuale']:.1f}%)", axis=1)

    if ordine:
        data["__ord"] = data["Risposta"].map({v: i for i, v in enumerate(ordine)})
        data = data.sort_values("__ord")
    else:
        data = data.sort_values("Conteggio", ascending=(orientamento == "h"))

    if orientamento == "h":
        fig = px.bar(data, x="Conteggio", y="Risposta", orientation="h",
                     text="label", color="Risposta",
                     color_discrete_sequence=PALETTE)
        fig.update_layout(yaxis_title=None, xaxis_title="Numero di risposte")
    else:
        fig = px.bar(data, x="Risposta", y="Conteggio",
                     text="label", color="Risposta",
                     color_discrete_sequence=PALETTE)
        fig.update_layout(xaxis_title=None, yaxis_title="Numero di risposte")

    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(title=titolo, showlegend=False, height=420,
                      margin=dict(t=60, r=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(nota_pct)


# ------------------------------------------------------------ WORD CLOUD ---

STOPWORDS_IT = {
    # articoli, preposizioni, congiunzioni, pronomi
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "a", "da",
    "in", "con", "su", "per", "tra", "fra", "e", "ed", "o", "od", "ma", "se",
    "che", "chi", "cui", "non", "più", "piu", "del", "della", "dello", "dei",
    "degli", "delle", "al", "allo", "alla", "ai", "agli", "alle", "dal",
    "dalla", "nel", "nella", "nei", "negli", "nelle", "sul", "sulla", "sui",
    "come", "anche", "quale", "quali", "qualche", "qualcosa", "questo",
    "questa", "questi", "queste", "quello", "quella", "mi", "ti", "si", "ci",
    "vi", "ne", "è", "sono", "sia", "essere", "ho", "hai", "ha", "mio", "mia",
    "miei", "mie", "tuo", "sua", "suo", "sue", "suoi", "loro", "molto",
    "tanto", "poi", "già", "gia", "ancora", "sempre", "tutto", "tutti",
    "tutte", "tipo", "etc", "ecc", "et", "similia", "ad", "oggi", "ora",
    # verbi/parole della domanda o di rumore
    "piacerebbe", "vorrei", "vedere", "conoscere", "approfondire", "imparare",
    "capire", "usare", "utilizzare", "creare", "fare", "gestire", "sapere",
    "interessano", "interesserebbe", "esiste", "potrei", "possa", "utile",
    "utili", "meglio", "bene", "no", "si", "sì", "saprei", "conosco", "idea",
    "idee", "particolare", "esempio", "magari", "forse", "spero", "dire",
    "so", "cosa", "cose", "momento", "adesso", "dopo", "corso", "però",
    "pero", "quindi", "essere", "avere", "mio", "nostro", "vostro",
    # termini generici legati alla domanda
    "tool", "tools", "applicazione", "applicazioni", "strumenti", "strumento",
    "ai", "ia", "intelligenza", "artificiale",
}

# Normalizzazione varianti/typo → forma canonica (multi-parola gestite prima)
NORMALIZZAZIONI = [
    (r"\bclaude\s*code\b", "claude_code"),
    (r"\bcloude\s*code\b", "claude_code"),
    (r"\bclaude\s*design\b", "claude_design"),
    (r"\bcloude\b", "claude"),
    (r"\bn8n\b", "n8n"),
    (r"\bnotebook\s*lm\b", "notebooklm"),
    (r"\bchat\s*gpt\b", "chatgpt"),
    (r"\bhiggsfild\b", "higgsfield"),
]

DISPLAY = {
    "claude_code": "Claude Code",
    "claude_design": "Claude Design",
    "chatgpt": "ChatGPT",
    "notebooklm": "NotebookLM",
}

FRASI_RUMORE = re.compile(
    r"^\s*(no|non\s+saprei|non\s+lo\s+so|non\s+ne\s+conosco|nessuno|niente|boh|/|-)\s*\.?\s*$",
    re.IGNORECASE,
)


def costruisci_frequenze(serie: pd.Series) -> Counter:
    counter = Counter()
    for testo in serie.dropna():
        testo = str(testo).strip()
        if not testo or FRASI_RUMORE.match(testo):
            continue
        t = testo.lower()
        for pattern, repl in NORMALIZZAZIONI:
            t = re.sub(pattern, repl, t)
        parole = re.findall(r"[a-zàèéìòù0-9_]+", t)
        viste_in_risposta = set()
        for w in parole:
            if len(w) < 2 or w in STOPWORDS_IT or w.isdigit():
                continue
            # conta ogni concetto una sola volta per risposta
            if w not in viste_in_risposta:
                counter[w] += 1
                viste_in_risposta.add(w)
    return counter


# ------------------------------------------------------------------- APP ---

st.title("☀️ Summer of AI — Analytics del sondaggio")

uploaded = st.sidebar.file_uploader("Carica il CSV delle submission", type="csv")
try:
    df = load_data(uploaded if uploaded is not None else DEFAULT_CSV)
except FileNotFoundError:
    st.info("Carica il file CSV dalla barra laterale per iniziare.")
    st.stop()

n_tot = len(df)
st.sidebar.metric("Risposte totali", n_tot)

# ---- KPI rapide
c1, c2, c3 = st.columns(3)
c1.metric("Rispondenti", n_tot)
eta = pd.to_numeric(df[COL_ETA], errors="coerce")
c2.metric("Età media", f"{eta.mean():.1f} anni" if eta.notna().any() else "n/d")
c3.metric("Hanno indicato un tool", int(df[COL_TOOL].dropna().astype(str).str.strip().ne("").sum()))

st.divider()

# ---- 1. Livello AI (ordinale)
livello = df[COL_LIVELLO].dropna().str.split(":").str[0].str.strip()
barre_con_percentuali(
    livello.value_counts(), f"📊 {COL_LIVELLO}",
    base_pct=livello.shape[0],
    nota_pct=f"% calcolata su {livello.shape[0]} risposte alla domanda.",
    orientamento="v", ordine=ORDINE_LIVELLO,
)

# ---- 2. Motivazione (multi-risposta)
motivi = df[COL_MOTIVO].apply(lambda r: estrai_opzioni(r, OPZIONI_MOTIVO))
n_risp_motivo = int(motivi.apply(len).gt(0).sum())
counts_motivo = pd.Series(Counter(m for lst in motivi for m in lst)).sort_values(ascending=False)
barre_con_percentuali(
    counts_motivo, f"🎯 {COL_MOTIVO}",
    base_pct=n_risp_motivo,
    nota_pct=f"Domanda multi-risposta: % calcolata su {n_risp_motivo} rispondenti (il totale può superare il 100%).",
)

# ---- 3. Ambito (multi-risposta, etichette abbreviate)
ambiti = df[COL_AMBITO].apply(lambda r: estrai_opzioni(r, OPZIONI_AMBITO))
n_risp_ambito = int(ambiti.apply(len).gt(0).sum())
counts_ambito = pd.Series(
    Counter(a.split(":")[0] for lst in ambiti for a in lst)
).sort_values(ascending=False)
barre_con_percentuali(
    counts_ambito, f"🧭 {COL_AMBITO}",
    base_pct=n_risp_ambito,
    nota_pct=f"Domanda multi-risposta: % calcolata su {n_risp_ambito} rispondenti (il totale può superare il 100%).",
)

# ---- 4. Età (fasce)
bins = [0, 18, 25, 35, 45, 55, 65, 120]
labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
fasce = pd.cut(eta.dropna(), bins=bins, labels=labels, right=False)
barre_con_percentuali(
    fasce.value_counts(), f"🎂 {COL_ETA} (per fascia)",
    base_pct=int(eta.notna().sum()),
    nota_pct=f"% calcolata su {int(eta.notna().sum())} risposte valide.",
    orientamento="v", ordine=labels,
)

# ---- 5. Occupazione
occup = df[COL_OCCUP].dropna()
barre_con_percentuali(
    occup.value_counts(), f"💼 {COL_OCCUP}",
    base_pct=occup.shape[0],
    nota_pct=f"% calcolata su {occup.shape[0]} risposte alla domanda.",
)

# ---- 6. Area di lavoro (top N + Altro)
area = df[COL_AREA].dropna().astype(str).str.strip()
area = area[area.ne("")]
vc_area = area.value_counts()
TOP_N = 10
if len(vc_area) > TOP_N:
    top = vc_area.head(TOP_N)
    altro = vc_area.iloc[TOP_N:].sum()
    vc_area = pd.concat([top, pd.Series({"Altro": altro})])
barre_con_percentuali(
    vc_area, f"🏢 {COL_AREA}",
    base_pct=area.shape[0],
    nota_pct=f"% calcolata su {area.shape[0]} risposte alla domanda. Le aree meno frequenti sono raggruppate in 'Altro'.",
)

st.divider()

# ---- 7. Word cloud tool
st.subheader(f"☁️ {COL_TOOL}")
freq = costruisci_frequenze(df[COL_TOOL])
freq_display = {DISPLAY.get(k, k.replace("_", " ").title() if "_" in k else k.capitalize()): v
                for k, v in freq.items()}

min_freq = st.slider("Frequenza minima per comparire nella cloud", 1, 10, 2)
freq_filtrata = {k: v for k, v in freq_display.items() if v >= min_freq}

if freq_filtrata:
    wc = WordCloud(width=1200, height=500, background_color="white",
                   colormap="viridis", prefer_horizontal=0.9,
                   max_words=80).generate_from_frequencies(freq_filtrata)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("🔝 Top 20 concetti più citati"):
        top20 = pd.Series(freq_filtrata).sort_values(ascending=False).head(20).reset_index()
        top20.columns = ["Concetto", "Citazioni"]
        st.dataframe(top20, use_container_width=True, hide_index=True)
else:
    st.warning("Nessun concetto supera la frequenza minima selezionata.")
