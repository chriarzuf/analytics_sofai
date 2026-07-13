# -*- coding: utf-8 -*-
"""
Summer of AI — Dashboard unica (Analytics + Clustering interpretabile)
Avvio locale:  streamlit run app.py
Streamlit Cloud: punta questo file come entrypoint.
Dipendenze: vedi requirements.txt
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import prince  # MCA per dati categorici
    HAS_PRINCE = True
except ImportError:
    HAS_PRINCE = False

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
COL_EMAIL = "e"

OPZ_MOTIVO = {
    "Produttività": "Per essere più produttivo",
    "Competitività": "Per essere più competitivo sul mercato",
    "Reinvenzione professionale": "Per reinventarmi professionalmente",
    "Curiosità": "Per pura curiosità",
}
OPZ_AMBITO = {
    "Tech": "Tech: per scrivere codice, creare software o gestire database",
    "Business": "Business: per gestire email, analizzare dati o scrivere report",
    "Creatività": "Creatività: per generare immagini e video o scrivere testi",
    "Studio": "Studio: per preparare esami, fare ricerche o imparare nuove lingue",
    "Hobby": "Hobby: per pianificare viaggi, gestire le finanze personali o progetti personali",
}
LIV_ORDER = ["Zero", "Base", "Intermedio", "Avanzato", "Master"]
LIV_MAP = {v: i for i, v in enumerate(LIV_ORDER)}
FASCE_ETA = ["<25", "25-34", "35-44", "45-54", "55+"]
ETA_BINS = [0, 25, 35, 45, 55, 120]

PALETTE = px.colors.qualitative.Set2


@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    return df


# ============================================================ ANALYTICS ====

def barre_con_percentuali(counts: pd.Series, titolo: str, base_pct: int,
                          nota_pct: str, orientamento: str = "h",
                          ordine: list | None = None):
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
                     text="label", color="Risposta", color_discrete_sequence=PALETTE)
        fig.update_layout(yaxis_title=None, xaxis_title="Numero di risposte")
    else:
        fig = px.bar(data, x="Risposta", y="Conteggio",
                     text="label", color="Risposta", color_discrete_sequence=PALETTE)
        fig.update_layout(xaxis_title=None, yaxis_title="Numero di risposte")

    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(title=titolo, showlegend=False, height=420, margin=dict(t=60, r=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(nota_pct)


STOPWORDS_IT = {
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
    "piacerebbe", "vorrei", "vedere", "conoscere", "approfondire", "imparare",
    "capire", "usare", "utilizzare", "creare", "fare", "gestire", "sapere",
    "interessano", "interesserebbe", "esiste", "potrei", "possa", "utile",
    "utili", "meglio", "bene", "no", "si", "sì", "saprei", "conosco", "idea",
    "idee", "particolare", "esempio", "magari", "forse", "spero", "dire",
    "so", "cosa", "cose", "momento", "adesso", "dopo", "corso", "però",
    "pero", "quindi", "avere", "nostro", "vostro",
    "tool", "tools", "applicazione", "applicazioni", "strumenti", "strumento",
    "ai", "ia", "intelligenza", "artificiale",
}
NORMALIZZAZIONI = [
    (r"\bclaude\s*code\b", "claude_code"),
    (r"\bcloude\s*code\b", "claude_code"),
    (r"\bclaude\s*design\b", "claude_design"),
    (r"\bcloude\b", "claude"),
    (r"\bnotebook\s*lm\b", "notebooklm"),
    (r"\bchat\s*gpt\b", "chatgpt"),
    (r"\bhiggsfild\b", "higgsfield"),
]
DISPLAY = {"claude_code": "Claude Code", "claude_design": "Claude Design",
           "chatgpt": "ChatGPT", "notebooklm": "NotebookLM"}
FRASI_RUMORE = re.compile(
    r"^\s*(no|non\s+saprei|non\s+lo\s+so|non\s+ne\s+conosco|nessuno|niente|boh|/|-)\s*\.?\s*$",
    re.IGNORECASE)


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
        viste = set()
        for w in parole:
            if len(w) < 2 or w in STOPWORDS_IT or w.isdigit() or w in viste:
                continue
            counter[w] += 1
            viste.add(w)
    return counter


def pagina_analytics(df: pd.DataFrame):
    n_tot = len(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rispondenti", n_tot)
    eta = pd.to_numeric(df[COL_ETA], errors="coerce")
    c2.metric("Età media", f"{eta.mean():.1f} anni" if eta.notna().any() else "n/d")
    c3.metric("Hanno indicato un tool",
              int(df[COL_TOOL].dropna().astype(str).str.strip().ne("").sum()))
    st.divider()

    livello = df[COL_LIVELLO].dropna().str.split(":").str[0].str.strip()
    barre_con_percentuali(livello.value_counts(), f"📊 {COL_LIVELLO}",
                          base_pct=livello.shape[0],
                          nota_pct=f"% calcolata su {livello.shape[0]} risposte alla domanda.",
                          orientamento="v", ordine=LIV_ORDER)

    def conta_multi(col, opzioni):
        estratte = df[col].apply(
            lambda r: [n for n, t in opzioni.items() if isinstance(r, str) and t in r])
        n_risp = int(estratte.apply(len).gt(0).sum())
        counts = pd.Series(Counter(m for lst in estratte for m in lst)).sort_values(ascending=False)
        return counts, n_risp

    counts_m, n_m = conta_multi(COL_MOTIVO, OPZ_MOTIVO)
    barre_con_percentuali(counts_m, f"🎯 {COL_MOTIVO}", base_pct=n_m,
                          nota_pct=f"Domanda multi-risposta: % su {n_m} rispondenti (il totale può superare il 100%).")

    counts_a, n_a = conta_multi(COL_AMBITO, OPZ_AMBITO)
    barre_con_percentuali(counts_a, f"🧭 {COL_AMBITO}", base_pct=n_a,
                          nota_pct=f"Domanda multi-risposta: % su {n_a} rispondenti (il totale può superare il 100%).")

    fasce = pd.cut(eta.dropna(), bins=ETA_BINS, labels=FASCE_ETA, right=False)
    barre_con_percentuali(fasce.value_counts(), f"🎂 {COL_ETA} (per fascia)",
                          base_pct=int(eta.notna().sum()),
                          nota_pct=f"% calcolata su {int(eta.notna().sum())} risposte valide.",
                          orientamento="v", ordine=FASCE_ETA)

    occup = df[COL_OCCUP].dropna()
    barre_con_percentuali(occup.value_counts(), f"💼 {COL_OCCUP}",
                          base_pct=occup.shape[0],
                          nota_pct=f"% calcolata su {occup.shape[0]} risposte alla domanda.")

    area = df[COL_AREA].dropna().astype(str).str.strip()
    area = area[area.ne("")]
    vc_area = area.value_counts()
    if len(vc_area) > 10:
        vc_area = pd.concat([vc_area.head(10), pd.Series({"Altro": vc_area.iloc[10:].sum()})])
    barre_con_percentuali(vc_area, f"🏢 {COL_AREA}", base_pct=area.shape[0],
                          nota_pct=f"% su {area.shape[0]} risposte. Aree meno frequenti raggruppate in 'Altro'.")

    st.divider()
    st.subheader(f"☁️ {COL_TOOL}")
    freq = costruisci_frequenze(df[COL_TOOL])
    freq_display = {DISPLAY.get(k, k.replace("_", " ").title() if "_" in k else k.capitalize()): v
                    for k, v in freq.items()}
    min_freq = st.slider("Frequenza minima per comparire nella cloud", 1, 10, 2)
    freq_f = {k: v for k, v in freq_display.items() if v >= min_freq}
    if freq_f:
        wc = WordCloud(width=1200, height=500, background_color="white",
                       colormap="viridis", prefer_horizontal=0.9,
                       max_words=80).generate_from_frequencies(freq_f)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("🔝 Top 20 concetti più citati"):
            top20 = pd.Series(freq_f).sort_values(ascending=False).head(20).reset_index()
            top20.columns = ["Concetto", "Citazioni"]
            st.dataframe(top20, use_container_width=True, hide_index=True)
    else:
        st.warning("Nessun concetto supera la frequenza minima selezionata.")


# ============================================================ CLUSTERING ===

@st.cache_data
def build_features(df: pd.DataFrame):
    """X = matrice numerica (solo per l'algoritmo).
    labels = risposte originali leggibili (per l'interpretazione)."""
    X = pd.DataFrame(index=df.index)
    labels = pd.DataFrame(index=df.index)

    for nome, testo in OPZ_MOTIVO.items():
        col = df[COL_MOTIVO].fillna("").str.contains(testo, regex=False).astype(int)
        X[f"Motivo: {nome}"] = col
        labels[f"Motivo: {nome}"] = col
    for nome, testo in OPZ_AMBITO.items():
        col = df[COL_AMBITO].fillna("").str.contains(testo, regex=False).astype(int)
        X[f"Ambito: {nome}"] = col
        labels[f"Ambito: {nome}"] = col

    liv_txt = df[COL_LIVELLO].str.split(":").str[0].str.strip()
    liv = liv_txt.map(LIV_MAP)
    X["Livello AI"] = liv.fillna(liv.median()) / 4
    labels["Livello AI"] = liv_txt.fillna("Non specificato")

    eta = pd.to_numeric(df[COL_ETA], errors="coerce")
    fascia_idx = pd.cut(eta, ETA_BINS, labels=range(5), right=False).astype(float)
    X["Fascia età"] = fascia_idx.fillna(fascia_idx.median()) / 4
    labels["Fascia età"] = pd.cut(eta, ETA_BINS, labels=FASCE_ETA,
                                  right=False).astype(str).replace("nan", "Non specificato")

    occ = df[COL_OCCUP].fillna("Non specificato")
    X = pd.concat([X, pd.get_dummies(occ, prefix="Occupazione").astype(int)], axis=1)
    labels["Occupazione"] = occ

    area = df[COL_AREA].fillna("Non specificato").astype(str).str.strip().replace("", "Non specificato")
    top = area.value_counts().head(8).index
    area = area.where(area.isin(top), "Altro")
    X = pd.concat([X, pd.get_dummies(area, prefix="Area").astype(int)], axis=1)
    labels["Area lavoro"] = area

    return X, labels


def _mca_variance(mca) -> np.ndarray:
    """Estrae la varianza spiegata gestendo le diverse versioni dell'API di prince."""
    if hasattr(mca, "percentage_of_variance_"):
        return np.asarray(mca.percentage_of_variance_) / 100.0
    if hasattr(mca, "explained_inertia_"):
        return np.asarray(mca.explained_inertia_)
    if hasattr(mca, "eigenvalues_"):
        ev = np.asarray(mca.eigenvalues_)
        return ev / ev.sum()
    return np.array([np.nan])


@st.cache_data
def compute_embedding(X: pd.DataFrame, method: str, n_comp: int, seed: int = 42):
    """Ritorna (embedding, varianza spiegata per componente).
    method: 'PCA' oppure 'MCA' (richiede prince; altrimenti fallback a PCA)."""
    n_comp = min(n_comp, X.shape[1])
    if method == "MCA" and HAS_PRINCE:
        # MCA lavora su categorie: converto ogni colonna in stringa
        # (le binarie 0/1 diventano categorie "0"/"1", le ordinali restano ordinabili)
        X_cat = X.astype(str)
        mca = prince.MCA(n_components=n_comp, random_state=seed)
        mca = mca.fit(X_cat)
        emb = mca.row_coordinates(X_cat).to_numpy()
        evr = _mca_variance(mca)[:n_comp]
        return emb, evr
    # PCA su one-hot standardizzato
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_comp, random_state=seed)
    emb = pca.fit_transform(Z)
    return emb, pca.explained_variance_ratio_


@st.cache_data
def run_clustering(X: pd.DataFrame, k: int, method: str, n_comp: int, seed: int = 42):
    emb, evr = compute_embedding(X, method, n_comp, seed)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    lab = km.fit_predict(emb)
    return lab, emb, silhouette_score(emb, lab), evr


@st.cache_data
def silhouette_curve(X: pd.DataFrame, method: str, n_comp: int, k_min=2, k_max=8, seed=42):
    emb, _ = compute_embedding(X, method, n_comp, seed)
    return pd.DataFrame([
        {"k": k, "silhouette": silhouette_score(
            emb, KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb))}
        for k in range(k_min, k_max + 1)
    ])


def pagina_clustering(df: pd.DataFrame):
    st.markdown(
        "Segmentazione basata su **tutte le risposte al sondaggio**. L'algoritmo lavora su "
        "una codifica numerica interna, ma **tutti i profili qui sotto mostrano le risposte "
        "originali**, così ogni cluster è pienamente interpretabile."
    )
    X, labels = build_features(df)

    # --- impostazioni algoritmo
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Impostazioni clustering")
    metodi = ["MCA (consigliato per dati categorici)", "PCA (su one-hot)"] if HAS_PRINCE \
        else ["PCA (su one-hot)"]
    if not HAS_PRINCE:
        st.sidebar.info("Libreria `prince` non installata: MCA non disponibile, uso PCA. "
                        "Aggiungi `prince` a requirements.txt per abilitarla.")
    metodo_sel = st.sidebar.radio("Riduzione dimensionale", metodi)
    method = "MCA" if metodo_sel.startswith("MCA") else "PCA"
    n_comp = st.sidebar.slider("Numero di componenti", 2, min(20, X.shape[1]), 12,
                               help="Più componenti = più varianza catturata, ma anche più rumore. "
                                    "12 è un buon compromesso per questi dati.")

    # --- scelta k
    st.subheader("1️⃣ Scelta del numero di cluster")
    curve = silhouette_curve(X, method, n_comp)
    best_k = int(curve.loc[curve["silhouette"].idxmax(), "k"])
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig = px.line(curve, x="k", y="silhouette", markers=True,
                      title="Silhouette score (più alto = cluster più separati)")
        fig.add_vline(x=best_k, line_dash="dash", line_color="green",
                      annotation_text=f"migliore: k={best_k}")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        k = st.slider("Numero di cluster (k)", 2, 8, best_k)
        st.caption("Il k suggerito massimizza la separazione statistica; "
                   "k più alti possono dare segmenti più azionabili.")

    lab, emb, sil, evr = run_clustering(X, k, method, n_comp)
    cluster_col = pd.Series([f"Cluster {c + 1}" for c in lab], index=df.index, name="Cluster")
    labels = labels.copy()
    labels["Cluster"] = cluster_col

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Metodo", method)
    m2.metric("Silhouette", f"{sil:.3f}")
    var_tot = np.nansum(evr)
    m3.metric("Varianza spiegata", f"{var_tot:.0%}" if not np.isnan(var_tot) else "n/d")
    m4.metric("Rispondenti", len(df))
    if method == "MCA":
        st.caption("Nota: nella MCA la % di inerzia spiegata è tipicamente bassa per costruzione "
                   "(correzione di Benzécri a parte) e non è confrontabile con la varianza PCA — "
                   "non allarmarti se il numero sembra piccolo.")

    # --- mappa + dimensioni
    st.subheader("2️⃣ Mappa e dimensione dei cluster")
    plot_df = pd.DataFrame({"PC1": emb[:, 0], "PC2": emb[:, 1], "Cluster": cluster_col,
                            "Livello": labels["Livello AI"],
                            "Occupazione": labels["Occupazione"],
                            "Fascia età": labels["Fascia età"]})
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster",
                     hover_data=["Livello", "Occupazione", "Fascia età"],
                     color_discrete_sequence=PALETTE, opacity=0.75)
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Passa il mouse sui punti per vedere le risposte originali della persona.")

    sizes = cluster_col.value_counts().sort_index()
    fig = px.bar(x=sizes.index, y=sizes.values, color=sizes.index,
                 color_discrete_sequence=PALETTE,
                 text=[f"{v} ({v / len(df):.1%})" for v in sizes.values],
                 title="Dimensione dei cluster")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Rispondenti", height=360)
    st.plotly_chart(fig, use_container_width=True)

    # --- profili interpretabili
    st.subheader("3️⃣ Profilo dei cluster — cosa hanno risposto le persone")
    CATEGORICHE = ["Livello AI", "Fascia età", "Occupazione", "Area lavoro"]
    BINARIE = [c for c in labels.columns if c.startswith(("Motivo:", "Ambito:"))]

    tab_id, tab_cat, tab_bin, tab_diff = st.tabs(
        ["🪪 Carta d'identità", "Variabili categoriche", "Motivazioni & Ambiti",
         "Cosa distingue ogni cluster"])

    with tab_id:
        st.caption("Sintesi in linguaggio naturale: risposta più frequente per ogni domanda "
                   "e tratti che caratterizzano il cluster.")
        for cl in sorted(labels["Cluster"].unique()):
            sub = labels[labels["Cluster"] == cl]
            righe = [f"### {cl} — {len(sub)} persone ({len(sub) / len(labels):.0%})"]
            for col in CATEGORICHE:
                moda = sub[col].mode().iloc[0]
                quota = (sub[col] == moda).mean()
                righe.append(f"- **{col}**: {moda} ({quota:.0%} del cluster)")
            top_bin = sub[BINARIE].mean().sort_values(ascending=False).head(3)
            righe.append("- **Motivazioni/ambiti prevalenti**: " + ", ".join(
                f"{c.split(': ')[1]} ({v:.0%})" for c, v in top_bin.items()))
            st.markdown("\n".join(righe))
            st.divider()

    with tab_cat:
        st.caption("Distribuzione % interna a ogni cluster (cluster di dimensioni diverse "
                   "sono così confrontabili). Le etichette sono le risposte originali.")
        ORDINI = {
            "Livello AI": LIV_ORDER + ["Non specificato"],
            "Fascia età": FASCE_ETA + ["Non specificato"],
        }
        for col in CATEGORICHE:
            ct = (labels.groupby("Cluster")[col].value_counts(normalize=True)
                  .rename("Quota").reset_index())
            ct["Quota"] *= 100
            ordine = ORDINI.get(col)
            if ordine:
                # mantieni solo le categorie presenti, nell'ordine naturale
                ordine = [v for v in ordine if v in ct[col].unique()]
            fig = px.bar(ct, x="Cluster", y="Quota", color=col, barmode="group",
                         text=ct["Quota"].map(lambda v: f"{v:.0f}%"), title=col,
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         category_orders={col: ordine} if ordine else None)
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(yaxis_title="% nel cluster", height=420)
            st.plotly_chart(fig, use_container_width=True)

    with tab_bin:
        prof = labels.groupby("Cluster")[BINARIE].mean().T * 100
        prof.index.name = "Variabile"
        fig = px.imshow(prof.round(0), text_auto=".0f", aspect="auto",
                        color_continuous_scale="YlGnBu",
                        title="% di rispondenti che hanno selezionato quella risposta, per cluster")
        fig.update_layout(height=520, coloraxis_colorbar_title="%")
        st.plotly_chart(fig, use_container_width=True)

    with tab_diff:
        st.caption("Scostamento (in punti %) rispetto alla media generale: le barre più lunghe "
                   "sono ciò che rende quel cluster diverso da tutti gli altri.")
        media_glob = labels[BINARIE].mean()
        for cl in sorted(labels["Cluster"].unique()):
            sub = labels[labels["Cluster"] == cl]
            diff = ((sub[BINARIE].mean() - media_glob) * 100).sort_values()
            dd = diff.reset_index()
            dd.columns = ["Risposta", "Scostamento"]
            fig = px.bar(dd, x="Scostamento", y="Risposta", orientation="h",
                         color="Scostamento", color_continuous_scale="RdBu_r",
                         range_color=[-40, 40],
                         text=dd["Scostamento"].map(lambda v: f"{v:+.0f} pt"),
                         title=f"{cl} vs media generale")
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(height=380, yaxis_title=None, showlegend=False,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # --- export con risposte originali
    st.subheader("4️⃣ Esporta assegnazioni (email + risposte + cluster)")
    export = pd.concat([
        df[COL_EMAIL].rename("email"),
        cluster_col,
        labels.drop(columns="Cluster"),
    ], axis=1)
    st.dataframe(export.head(20), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Scarica CSV completo",
                       export.to_csv(index=False).encode("utf-8-sig"),
                       file_name="clusters_summer_of_ai.csv", mime="text/csv")


# ================================================================== MAIN ===

st.title("☀️ Summer of AI — Dashboard")

uploaded = st.sidebar.file_uploader("Carica il CSV delle submission", type="csv")
try:
    df = load_data(uploaded if uploaded is not None else DEFAULT_CSV)
except FileNotFoundError:
    st.info("Carica il file CSV dalla barra laterale per iniziare.")
    st.stop()

st.sidebar.metric("Risposte totali", len(df))
pagina = st.sidebar.radio("Sezione", ["📊 Analytics sondaggio", "🧩 Clustering"])

if pagina.startswith("📊"):
    pagina_analytics(df)
else:
    pagina_clustering(df)
