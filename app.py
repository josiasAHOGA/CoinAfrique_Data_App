"""
Application de Web Scraping - CoinAfrique
Projet 2 - Master 1 IA, DIT Senegal
Auteur: Josias AHOGA

Ce projet permet de scraper les annonces sur CoinAfrique (vetements et chaussures)
et d'analyser les donnees avec des graphiques.
"""

# ===========================================
# IMPORTATION DES BIBLIOTHEQUES
# ===========================================

import streamlit as st          # pour l'interface web
import pandas as pd             # pour manipuler les tableaux de donnees
import numpy as np              # pour les calculs
import re                       # pour les expressions regulieres
import time                     # pour les pauses entre requetes
from requests import get        # pour envoyer des requetes HTTP
from bs4 import BeautifulSoup   # pour parser le HTML
import matplotlib.pyplot as plt # pour les graphiques
import seaborn as sns           # pour les graphiques plus jolis


# ===========================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ===========================================

st.set_page_config(
    page_title="CoinAfrique - Scraping & Analyse",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================
# STYLE CSS POUR L'APPLICATION
# ===========================================

st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            color: #1f2937;
        }
        .subtitle {
            text-align: center;
            font-size: 1rem;
            color: #4b5563;
            margin-bottom: 1.2rem;
        }
        .box {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 14px 16px;
            margin: 10px 0 16px 0;
        }
        .box h3 {
            margin: 0 0 8px 0;
            color: #111827;
        }
        .muted {
            color: #6b7280;
            font-size: 0.95rem;
        }
        section[data-testid="stSidebar"] {
            background: #0b1220;
        }
        section[data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================
# TITRE DE L'APPLICATION
# ===========================================

st.markdown("<div class='title'>COINAFRIQUE - DATA APP</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Scraping, nettoyage et analyse des donnees</div>", unsafe_allow_html=True)


# ===========================================
# FONCTIONS UTILITAIRES
# ===========================================

@st.cache_data
def to_csv_bytes(df):
    """Convertit un dataframe en CSV pour le telechargement"""
    return df.to_csv(index=False).encode("utf-8-sig")


def clean_price(price_str):
    """
    Nettoie le prix texte et le convertit en nombre
    Exemple: "5 000 CFA" devient 5000.0
    """
    if pd.isna(price_str):
        return np.nan

    s = str(price_str).lower()

    # si c'est "prix sur demande"
    if "demande" in s:
        return np.nan

    # extraction des chiffres avec regex
    numbers = re.findall(r"\d+(?:\s*\d+)*", s)
    
    if not numbers:
        return np.nan

    try:
        return float(int(numbers[0].replace(" ", "")))
    except:
        return np.nan


def split_location(adresse):
    """
    Separe l'adresse en quartier et ville
    Exemple: "Medina, Dakar, Senegal" -> quartier="Medina", ville="Senegal"
    """
    if pd.isna(adresse) or str(adresse).strip() == "":
        return "Inconnu", "Inconnue"

    parts = [p.strip() for p in str(adresse).split(",") if p.strip()]
    quartier = parts[0] if parts else "Inconnu"
    ville = parts[-1] if parts else "Inconnue"

    return quartier, ville


def categorize_product(product_type):
    """Categorise le produit selon son type"""
    if pd.isna(product_type):
        return "Autre"

    s = str(product_type).lower()

    if any(k in s for k in ["pantalon", "jean", "short", "bermuda", "jupe"]):
        return "Bas"
    if any(k in s for k in ["t-shirt", "tshirt", "chemise", "polo", "pull", "veste"]):
        return "Haut"
    if any(k in s for k in ["chauss", "basket", "sandale", "jordan", "nike"]):
        return "Chaussures"
    if any(k in s for k in ["ensemble", "costume", "tenue"]):
        return "Ensemble"

    return "Autre"


def clean_dataframe(df):
    """Nettoie le dataframe et ajoute les colonnes utiles"""
    dfc = df.copy()

    # renommer les colonnes
    rename_map = {}
    for col in dfc.columns:
        low = col.lower()
        if low in ["type produit", "type_produit", "type", "type_habits", "types_chaussures"]:
            rename_map[col] = "Type_Produit"
        if low in ["prix", "price"]:
            rename_map[col] = "Prix"
        if low in ["adresse", "location", "lieu"]:
            rename_map[col] = "adresse"
        if low in ["image_lien", "image", "image_url"]:
            rename_map[col] = "Image_URL"

    if rename_map:
        dfc = dfc.rename(columns=rename_map)

    # supprimer colonnes web scraper
    for c in ["web_scraper_order", "web_scraper_start_url"]:
        if c in dfc.columns:
            dfc = dfc.drop(columns=c)

    # convertir le prix
    if "Prix" in dfc.columns:
        dfc["Prix_Numerique"] = dfc["Prix"].apply(clean_price)

    # separer l'adresse
    if "adresse" in dfc.columns:
        quartiers, villes = [], []
        for adr in dfc["adresse"].tolist():
            q, v = split_location(adr)
            quartiers.append(q)
            villes.append(v)
        dfc["Quartier"] = quartiers
        dfc["Ville"] = villes

    # categoriser les produits
    if "Type_Produit" in dfc.columns:
        dfc["Categorie"] = dfc["Type_Produit"].apply(categorize_product)

    # supprimer doublons
    dfc = dfc.drop_duplicates()

    return dfc


# ===========================================
# FONCTION DE SCRAPING
# ===========================================

def make_absolute_url(url):
    """Transforme URL relative en absolue"""
    if not url:
        return ""
    if url.startswith("http"):
        return url
    return f"https://sn.coinafrique.com{url}"


def scrape_coinafrique_category(base_url, pages, source_label):
    """
    Scrape les annonces d'une categorie sur plusieurs pages
    Retourne un dataframe avec les donnees
    """
    data = []
    progress = st.progress(0)
    status = st.empty()

    # user-agent pour simuler un navigateur
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    for p in range(1, pages + 1):
        status.write(f"Page {p}/{pages}...")
        page_url = f"{base_url}?page={p}"

        try:
            resp = get(page_url, headers=headers, timeout=15)
            
            if resp.status_code != 200:
                progress.progress(p / pages)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            
            # chercher les liens d'annonces
            anchors = soup.find_all("a", href=re.compile(r"/annonce/"))

            for a in anchors:
                try:
                    annonce_url = make_absolute_url(a.get("href", ""))

                    # titre du produit
                    title_tag = a.find(["h2", "p", "span"], string=True)
                    type_produit = title_tag.get_text(strip=True) if title_tag else "N/A"

                    # prix
                    price_tag = a.find(string=re.compile(r"CFA|F\s*CFA|\d"))
                    prix_text = "Prix sur demande"
                    if price_tag:
                        prix_text = str(price_tag).strip()
                        if len(prix_text) > 40:
                            prix_text = "Prix sur demande"

                    # adresse
                    adr_tag = a.find(attrs={"class": re.compile(r"location|adress", re.I)})
                    adresse_text = adr_tag.get_text(" ", strip=True) if adr_tag else ""

                    # image
                    img = a.find("img")
                    image_url = ""
                    if img:
                        image_url = img.get("src") or img.get("data-src") or ""
                        image_url = make_absolute_url(image_url)

                    data.append({
                        "Type_Produit": type_produit,
                        "Prix": prix_text,
                        "adresse": adresse_text,
                        "Image_URL": image_url,
                        "Annonce_URL": annonce_url,
                        "Source": source_label,
                    })

                except:
                    continue

            progress.progress(p / pages)
            time.sleep(0.8)  # pause entre les pages

        except:
            progress.progress(p / pages)
            continue

    status.empty()
    progress.empty()

    return pd.DataFrame(data)


# ===========================================
# SIDEBAR - MENU DE NAVIGATION
# ===========================================

st.sidebar.markdown("## CoinAfrique Data App")
st.sidebar.markdown("**Auteur : Josias AHOGA**")
st.sidebar.markdown("Master 1 IA - DIT Senegal")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Scraping (BeautifulSoup)", "Charger CSV (Web Scraper)", "Dashboard", "Feedback", "A propos"],
)


# ===========================================
# PAGE 1 : SCRAPING
# ===========================================

if page == "Scraping (BeautifulSoup)":
    st.markdown("""
        <div class='box'>
            <h3>Scraper CoinAfrique en direct</h3>
            <div class='muted'>Selectionnez les categories et le nombre de pages a scraper.</div>
        </div>
    """, unsafe_allow_html=True)

    # les 4 categories du projet
    url_map = {
        "Vetements Homme": ("https://sn.coinafrique.com/categorie/vetements-homme", "vetements_homme"),
        "Chaussures Homme": ("https://sn.coinafrique.com/categorie/chaussures-homme", "chaussures_homme"),
        "Vetements Enfants": ("https://sn.coinafrique.com/categorie/vetements-enfants", "vetements_enfants"),
        "Chaussures Enfants": ("https://sn.coinafrique.com/categorie/chaussures-enfants", "chaussures_enfants"),
    }

    # variables collectees
    st.markdown("#### Variables collectees")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.write("- V1 : Type_Produit")
        st.write("- V2 : Prix")
    with col_v2:
        st.write("- V3 : adresse")
        st.write("- V4 : Image_lien")

    st.markdown("---")

    # selection des categories
    st.markdown("#### Etape 1 : Choisir les categories")
    
    col_cat1, col_cat2 = st.columns(2)
    with col_cat1:
        check_vh = st.checkbox("Vetements Homme", value=True)
        check_ve = st.checkbox("Vetements Enfants", value=False)
    with col_cat2:
        check_ch = st.checkbox("Chaussures Homme", value=False)
        check_ce = st.checkbox("Chaussures Enfants", value=False)

    selected_categories = []
    if check_vh: selected_categories.append("Vetements Homme")
    if check_ch: selected_categories.append("Chaussures Homme")
    if check_ve: selected_categories.append("Vetements Enfants")
    if check_ce: selected_categories.append("Chaussures Enfants")

    st.markdown("---")

    # nombre de pages
    st.markdown("#### Etape 2 : Nombre de pages")
    pages = st.slider("Pages par categorie", min_value=1, max_value=50, value=5)

    st.markdown("---")

    # estimation du temps
    st.markdown("#### Estimation du temps")
    
    nb_cat = len(selected_categories)
    
    if nb_cat == 0:
        st.warning("Selectionnez au moins une categorie.")
    else:
        temps_total = nb_cat * pages * 1.2  # 1.2 sec par page
        minutes = int(temps_total // 60)
        secondes = int(temps_total % 60)
        annonces_est = nb_cat * pages * 25

        col1, col2, col3 = st.columns(3)
        col1.metric("Categories", nb_cat)
        col2.metric("Pages totales", nb_cat * pages)
        if minutes > 0:
            col3.metric("Temps estime", f"{minutes} min {secondes} sec")
        else:
            col3.metric("Temps estime", f"{secondes} sec")
        
        st.info(f"Environ {annonces_est} annonces seront collectees.")

        st.markdown("---")

        # bouton scraping
        if st.button("Lancer le scraping", type="primary", use_container_width=True):
            
            all_data = pd.DataFrame()
            progress_global = st.progress(0)
            status_global = st.empty()
            start_time = time.time()

            for idx, cat in enumerate(selected_categories):
                base_url, label = url_map[cat]
                status_global.markdown(f"**{cat}** ({idx+1}/{nb_cat})")
                
                raw = scrape_coinafrique_category(base_url, pages, label)
                
                if not raw.empty:
                    raw["Source"] = cat
                    all_data = pd.concat([all_data, raw], ignore_index=True)
                    st.write(f"- {cat} : {len(raw)} annonces")
                
                progress_global.progress((idx + 1) / nb_cat)

            temps_reel = time.time() - start_time
            progress_global.empty()
            status_global.empty()

            if all_data.empty:
                st.error("Aucune donnee collectee.")
            else:
                clean = clean_dataframe(all_data)
                st.session_state["data"] = clean
                
                st.success("Scraping termine !")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Annonces", len(clean))
                col2.metric("Categories", nb_cat)
                col3.metric("Pages", nb_cat * pages)
                col4.metric("Temps", f"{int(temps_reel)} sec")
                
                st.dataframe(clean, use_container_width=True, height=400)
                
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button("Telecharger brut", to_csv_bytes(all_data), "data_raw.csv", "text/csv")
                with col_dl2:
                    st.download_button("Telecharger nettoye", to_csv_bytes(clean), "data_clean.csv", "text/csv")


# ===========================================
# PAGE 2 : CHARGER CSV
# ===========================================

elif page == "Charger CSV (Web Scraper)":
    st.markdown("""
        <div class='box'>
            <h3>Charger les fichiers CSV</h3>
            <div class='muted'>Fichiers issus de Web Scraper.</div>
        </div>
    """, unsafe_allow_html=True)

    csv_files = {
        "Vetements Homme": "vetement_homme.csv",
        "Chaussures Homme": "coinafrique_chaussures_homme__3_.csv",
        "Vetements Femme": "Coinafrique_vetement_femme__3_.csv",
        "Chaussures Femme": "coinafrique_chaussure_femme__3_.csv",
    }

    st.markdown("### Fichiers disponibles")
    for name, path in csv_files.items():
        st.write(f"- {name} : {path}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        btn_all = st.button("Charger TOUT", use_container_width=True)
        btn_vet = st.button("Vetements seulement", use_container_width=True)
    with col2:
        btn_homme = st.button("Categories Homme", use_container_width=True)
        btn_chau = st.button("Chaussures seulement", use_container_width=True)

    selected = []
    if btn_all: selected = list(csv_files.keys())
    elif btn_vet: selected = ["Vetements Homme", "Vetements Femme"]
    elif btn_chau: selected = ["Chaussures Homme", "Chaussures Femme"]
    elif btn_homme: selected = ["Vetements Homme", "Chaussures Homme"]

    if selected:
        frames = []
        for name in selected:
            try:
                df = pd.read_csv(csv_files[name], encoding="utf-8-sig")
                df["Source"] = name
                frames.append(clean_dataframe(df))
                st.write(f"- {name} : {len(df)} lignes")
            except Exception as e:
                st.error(f"Erreur {name} : {e}")

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            st.session_state["data"] = combined
            st.success(f"Total : {len(combined)} lignes")
            st.dataframe(combined, use_container_width=True, height=350)
            st.download_button("Telecharger", to_csv_bytes(combined), "data_combined.csv", "text/csv")


# ===========================================
# PAGE 3 : DASHBOARD
# ===========================================

elif page == "Dashboard":
    st.markdown("""
        <div class='box'>
            <h3>Dashboard Analytique</h3>
            <div class='muted'>Analyse des donnees collectees.</div>
        </div>
    """, unsafe_allow_html=True)

    if "data" not in st.session_state or st.session_state["data"].empty:
        st.info("Aucune donnee. Allez sur Scraping ou Charger CSV.")
    else:
        df = st.session_state["data"].copy()
        
        # couleurs par categorie
        colors_cat = {
            "Vetements Homme": "#2E86AB", "Chaussures Homme": "#A23B72",
            "Vetements Enfants": "#F18F01", "Chaussures Enfants": "#C73E1D",
            "Vetements Femme": "#F18F01", "Chaussures Femme": "#C73E1D"
        }
        sns.set_style("whitegrid")

        # filtrer prix aberrants
        df_clean = df.copy()
        if "Prix_Numerique" in df_clean.columns:
            df_clean = df_clean[(df_clean["Prix_Numerique"] >= 500) & (df_clean["Prix_Numerique"] <= 200000)]

        # --- SECTION 1 : VUE GLOBALE ---
        st.markdown("## Vue Globale")
        
        if "Source" in df.columns:
            source_counts = df["Source"].value_counts()
            cols = st.columns(len(source_counts))
            for i, (src, cnt) in enumerate(source_counts.items()):
                cols[i].metric(src, f"{cnt} annonces")
            
            # graphique repartition
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            colors = [colors_cat.get(s, "#666") for s in source_counts.index]
            
            bars = ax1.bar(source_counts.index, source_counts.values, color=colors)
            for bar, val in zip(bars, source_counts.values):
                ax1.text(bar.get_x() + bar.get_width()/2, val + 10, str(val), ha='center', fontweight='bold')
            ax1.set_ylabel("Nombre d'annonces")
            ax1.set_title("Volume par categorie")
            ax1.tick_params(axis='x', rotation=15)
            
            ax2.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', colors=colors)
            ax2.set_title("Repartition")
            
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()

        # --- SECTION 2 : PRIX ---
        st.markdown("---")
        st.markdown("## Analyse des Prix")
        
        if "Prix_Numerique" in df_clean.columns and "Source" in df_clean.columns:
            stats = df_clean.groupby("Source")["Prix_Numerique"].agg(["count", "mean", "median"]).round(0)
            
            cols = st.columns(len(stats))
            for i, (src, row) in enumerate(stats.iterrows()):
                cols[i].metric(src, f"{row['median']:,.0f} CFA")
            
            # boxplot
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            df_clean.boxplot(column="Prix_Numerique", by="Source", ax=ax2, patch_artist=True)
            ax2.set_ylabel("Prix (CFA)")
            ax2.set_title("Distribution des prix")
            plt.suptitle("")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
            
            # moyenne vs mediane
            fig3, ax3 = plt.subplots(figsize=(12, 5))
            x = np.arange(len(stats))
            ax3.bar(x - 0.2, stats["mean"], 0.4, label="Moyenne", color="#2E86AB")
            ax3.bar(x + 0.2, stats["median"], 0.4, label="Mediane", color="#F18F01")
            ax3.set_xticks(x)
            ax3.set_xticklabels(stats.index, rotation=15)
            ax3.set_ylabel("Prix (CFA)")
            ax3.legend()
            ax3.set_title("Prix Moyen vs Median")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        # --- SECTION 3 : ANALYSE CROISEE ---
        st.markdown("---")
        st.markdown("## Analyse Croisee")
        
        if "Source" in df_clean.columns:
            df_clean["Type"] = df_clean["Source"].apply(lambda x: "Vetements" if "Vetement" in x else "Chaussures")
            df_clean["Cible"] = df_clean["Source"].apply(lambda x: "Homme" if "Homme" in x else ("Enfants" if "Enfants" in x else "Femme"))
            
            fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))
            
            type_med = df_clean.groupby("Type")["Prix_Numerique"].median()
            ax4a.bar(type_med.index, type_med.values, color=["#2E86AB", "#A23B72"])
            ax4a.set_ylabel("Prix Median (CFA)")
            ax4a.set_title("Vetements vs Chaussures")
            
            cible_med = df_clean.groupby("Cible")["Prix_Numerique"].median()
            ax4b.bar(cible_med.index, cible_med.values, color=["#F18F01", "#C73E1D", "#95C623"][:len(cible_med)])
            ax4b.set_ylabel("Prix Median (CFA)")
            ax4b.set_title("Par Cible")
            
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()
            
            # heatmap
            pivot = df_clean.pivot_table(values="Prix_Numerique", index="Type", columns="Cible", aggfunc="median")
            fig5, ax5 = plt.subplots(figsize=(8, 4))
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax5)
            ax5.set_title("Prix Median (Type x Cible)")
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close()

        # --- SECTION 4 : TOP PRODUITS ---
        st.markdown("---")
        st.markdown("## Top Produits")
        
        if "Type_Produit" in df.columns and "Source" in df.columns:
            sources = df["Source"].unique()[:4]
            fig6, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for i, src in enumerate(sources):
                top = df[df["Source"] == src]["Type_Produit"].value_counts().head(7)
                color = colors_cat.get(src, "#666")
                axes[i].barh(top.index[::-1], top.values[::-1], color=color)
                axes[i].set_title(src)
            
            plt.tight_layout()
            st.pyplot(fig6)
            plt.close()

        # --- SECTION 5 : QUARTIERS ---
        st.markdown("---")
        st.markdown("## Top Quartiers")
        
        if "Quartier" in df.columns:
            top_q = df["Quartier"].value_counts().head(10)
            fig7, ax7 = plt.subplots(figsize=(10, 6))
            ax7.barh(top_q.index[::-1], top_q.values[::-1], color=sns.color_palette("viridis", 10)[::-1])
            ax7.set_xlabel("Nombre d'annonces")
            ax7.set_title("Top 10 Quartiers")
            plt.tight_layout()
            st.pyplot(fig7)
            plt.close()

        # --- SECTION 6 : TABLEAU ---
        st.markdown("---")
        st.markdown("## Tableau Recapitulatif")
        
        if "Source" in df_clean.columns:
            recap = df_clean.groupby("Source")["Prix_Numerique"].agg(["count", "min", "max", "mean", "median"]).round(0)
            recap.columns = ["Nb", "Min", "Max", "Moyenne", "Mediane"]
            st.dataframe(recap, use_container_width=True)

        # donnees
        st.markdown("---")
        st.markdown("## Donnees")
        st.dataframe(df, use_container_width=True, height=400)


# ===========================================
# PAGE 4 : FEEDBACK
# ===========================================

elif page == "Feedback":
    st.markdown("""
        <div class='box'>
            <h3>Evaluation de l'application</h3>
            <div class='muted'>Votre avis nous aide a ameliorer cette application.</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Donnez votre avis")
    
    st.write("""
    Merci d'avoir utilise cette application. Pour nous aider a l'ameliorer, 
    vous pouvez remplir un court formulaire d'evaluation.
    
    Le formulaire contient quelques questions sur :
    - La facilite d'utilisation
    - La qualite des donnees collectees
    - Les fonctionnalites du dashboard
    - Vos suggestions d'amelioration
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Acceder au formulaire d'evaluation", type="primary", use_container_width=True):
            st.markdown('<meta http-equiv="refresh" content="0; url=https://ee.kobotoolbox.org/x/yqgh0OmS">', unsafe_allow_html=True)


# ===========================================
# PAGE 5 : A PROPOS
# ===========================================

elif page == "A propos":
    st.markdown("""
        <div class='box'>
            <h3>A propos de l'application</h3>
            <div class='muted'>Informations sur le projet et son auteur.</div>
        </div>
    """, unsafe_allow_html=True)

    # informations auteur
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Auteur")
        st.write("**Josias AHOGA**")
        st.write("Master 1 Intelligence Artificielle")
        st.write("Dakar Institute of Technology")
        st.write("Senegal")
    
    with col2:
        st.markdown("### Technologies utilisees")
        
        tech_data = {
            "Composant": ["Interface web", "Scraping", "Donnees", "Visualisation"],
            "Technologie": ["Streamlit", "BeautifulSoup + Requests", "Pandas + NumPy", "Matplotlib + Seaborn"],
            "Role": [
                "Creation de l'application interactive",
                "Extraction des donnees depuis CoinAfrique",
                "Manipulation et nettoyage des donnees",
                "Graphiques et tableaux de bord"
            ]
        }
        st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    
    # fonctionnalites
    st.markdown("### Fonctionnalites de l'application")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown("**Scraping en direct**")
        st.write("Collecte les annonces directement depuis le site CoinAfrique avec BeautifulSoup.")
        
        st.markdown("**Chargement CSV**")
        st.write("Importe les fichiers CSV generes par l'extension Web Scraper.")
    
    with col_f2:
        st.markdown("**Dashboard analytique**")
        st.write("Visualise les donnees avec des graphiques interactifs et des statistiques.")
        
        st.markdown("**Export des donnees**")
        st.write("Telecharge les donnees brutes ou nettoyees au format CSV.")

    st.markdown("---")
    
    # variables
    st.markdown("### Variables collectees")
    
    var_data = {
        "Variable": ["V1", "V2", "V3", "V4"],
        "Nom": ["Type_Produit", "Prix", "Adresse", "Image_URL"],
        "Description": [
            "Type d'habit ou de chaussure",
            "Prix en francs CFA",
            "Localisation (quartier, ville)",
            "Lien vers l'image du produit"
        ]
    }
    st.dataframe(pd.DataFrame(var_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    
    # contact
    st.markdown("### Projet academique")
    st.write("Cette application a ete developpee dans le cadre du cours de collecte de donnees.")
    st.write("Annee academique 2024-2025")
