import streamlit as st
import pandas as pd
import numpy as np
import warnings
import zipfile
import pickle
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from rapidfuzz import process
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup
import requests
navigator = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1)'
url_base = 'https://www.imdb.com'
url_base_title = 'https://www.imdb.com/fr/title/'

import pickle


# FONCTIONS

# Fonction WebScrapping

def info_films(id):

    if not id:
        return None

    # Valeurs par défaut
    lien_trailer = "Aucune bande-annonce disponible"
    lien_affiche = "Aucune affiche disponible"
    liste_acteurs = ["Acteur inconnu"]
    dico_photos_final = {}
    realisateur = "Inconnu"
    resume = "Résumé non disponible"

    url_base = "https://www.imdb.com"
    url_title = f"https://www.imdb.com/fr/title/{id}"

    # --- 1 seule requête HTTP
    try:
        html = requests.get(url_title, headers={"User-Agent": navigator}).content
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return lien_trailer, lien_affiche, liste_acteurs, dico_photos_final, realisateur, resume

    # --- TRAILER
    lien_trailer = "Aucune bande-annonce disponible"

    trailer_tag = soup.find(
        "a",
        class_="ipc-lockup-overlay ipc-focusable ipc-focusable--constrained",
        href=True
    )

    if trailer_tag and "video" in trailer_tag["href"]:
        lien_trailer = "https://www.imdb.com" + trailer_tag["href"]


    # --- AFFICHE (Poster principal)
    poster = soup.find("img", {"class": "ipc-image"})
    if poster and poster.get("src"):
        lien_affiche = poster["src"]

    # --- ACTEURS (via data-testid)
    acteurs = [
        a.get_text(strip=True)
        for a in soup.find_all("a", attrs={"data-testid": "title-cast-item__actor"})
    ]
    if acteurs:
        acteurs = list(dict.fromkeys(acteurs))  # supprime doublons, garde l’ordre
        liste_acteurs = acteurs[:4]

    # --- PHOTOS ACTEURS
    image_tags = soup.find_all("img", class_="ipc-image")

    photos_map = {
        img.get("alt", "").strip(): img.get("src")
        for img in image_tags
        if img.get("alt") and img.get("src")
    }

    dico_photos_final = {
        acteur: photos_map.get(acteur, None)
        for acteur in liste_acteurs
        if acteur in photos_map
    }

    # --- RÉALISATEUR
    realisateur = "Inconnu"
    realisateur_tag = soup.find(
        "a",
        class_="ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link"
    )

    if realisateur_tag:
        realisateur = realisateur_tag.get_text(strip=True)


    # --- RÉSUMÉ
    resume_tag = soup.find("span", attrs={"data-testid": "plot-xs_to_m"})
    if resume_tag:
        resume = resume_tag.get_text(strip=True)

    return (
        lien_trailer,
        lien_affiche,
        liste_acteurs,
        dico_photos_final,
        realisateur,
        resume
    )



def encodage_X(X, type, poids):
    from sklearn.preprocessing import StandardScaler
    index = X.index
    X_num = X.select_dtypes('number')

    if type == 'standard':
        from sklearn.preprocessing import StandardScaler
        SN = StandardScaler()
        X_num_SN = pd.DataFrame(SN.fit_transform(X_num), columns=X_num.columns)

    else:
        from sklearn.preprocessing import MinMaxScaler
        SN = MinMaxScaler()
        X_num_SN = pd.DataFrame(SN.fit_transform(X_num), columns=X_num.columns)

    X_num_SN = X_num_SN.mul(poids, axis = 1)
    X_encoded = X_num_SN

    X_encoded = X_encoded.dropna()

    return X_encoded, SN

# FONCTION 2

def evaluate_k(X_encoded, k_range):

    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans

    avg_distances = []
    silhouette_scores = []

    for k in k_range:
        from sklearn.neighbors import NearestNeighbors
        model = NearestNeighbors(n_neighbors=k)
        model.fit(X_encoded)
        distances, _ = model.kneighbors(X_encoded)
        avg_distances.append(np.mean(distances))

        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_encoded)
        if k > 1:
            silhouette_scores.append(silhouette_score(X_encoded, clusters))
        else:
            silhouette_scores.append(0)

    return avg_distances, silhouette_scores

    # FONCTION 3

def encodage_predict(df_a_predire, SN, poids, X_encoded):
    X_num = df_a_predire.select_dtypes('number')

    X_num_SN = pd.DataFrame(SN.transform(X_num), columns=X_num.columns).reset_index(drop=True)
    X_num_SN = X_num_SN.mul(poids, axis = 1)
    
    X_encoded_predire = X_num_SN

    df_predict = X_encoded_predire

    df_final = pd.DataFrame(columns=X_encoded.columns)

    df_final = df_final.reindex(index=df_predict.index)

    df_final = df_final.fillna(False)

    for column in df_predict.columns:
        if column in X_encoded.columns:
            df_final[column] = df_predict[column]

    return df_final

# FONCTION 4
def pokemons_similaires(X, film_id, model, SN, poids, X_encoded, df):

    if film_id not in X['film_id_out_KNN'].values:
        return f"Le film {film_id} n'est pas dans le dataset."

    pokemon = X[X['film_id_out_KNN'] == film_id]

    caract_pokemon = X[X['film_id_out_KNN'] == film_id]

    caract_pokemon_encoded = encodage_predict(caract_pokemon, SN, poids, X_encoded)

    distances, indices = model.kneighbors(caract_pokemon_encoded)

    return df.iloc[indices[0]].reset_index(drop=True)

# Import des données

df = pd.read_csv("Codes/P2_G5_films.csv.gz", compression='gzip')

df = df.rename({'title_final_out_KNN' : 'title_out_KNN'}, axis = 1)

df['total_title_out_KNN'] = df['total_title_out_KNN'].apply(lambda x : x.lower())
# CHOIX DES CARACTERISTIQUES

caracteristiques = []

for element in df.columns:
    if 'out_KNN' not in element:
        caracteristiques.append(element)

caracteristiques_num = []

for element in df.select_dtypes(include = 'number').columns:
    if 'out_KNN' not in element:
        caracteristiques_num.append(element)

caracteristiques = [col for col in df.columns if 'out_KNN' not in col]
caracteristiques_num = [col for col in df.select_dtypes(include='number').columns if 'out_KNN' not in col]


# METTRE UNIQUEMENT POUR LES COLONNES NUMERIQUES

poids_list = pd.DataFrame(columns = caracteristiques_num, index = ['poids'])

colonne_cle = 10
tres_important = 7
important = 4
bof = 1
rien = 0

poids = {
 'popularity_final' : colonne_cle,
 'year_final' : bof,
 'Decennie' : tres_important,
 'runtime_final' : rien,
 'vote_exact_final' : important,
 'vote_arrondi_final' : colonne_cle,
 'vote_count_final' : tres_important,
 'prod_US' : important,
 'prod_FR' : important
}

for element in df.select_dtypes(include = 'number').columns:
    if "production_companies_name" in element:
        poids.update({element : important})
    elif "acteur_{" in element:
        poids.update({element : colonne_cle})
    elif "realisateurs_" in element:
        poids.update({element : colonne_cle})
    elif "genre_" in element:
        poids.update({element : colonne_cle})
    elif 'debut_titre_critere_' in element:
        poids.update({element : colonne_cle})
    # else:
    #      poids.update({element : rien})



for element in poids_list.columns:
    if element not in poids.keys():
        poids.update({element : rien})

# Import CSS

def local_css(styles):
    with open(styles) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("streamlit/styles.css")

# --------------


st.html("<h1>What the Rock'mendation?</h1>")

st.html("<p>Bienvenu sur notre service de recommandation de films. Choisissez un film et laissez-vous guider...</p>")


# Sélectionner le film

df_recherche = df.copy()
df_recherche4 = df_recherche.copy()

choix_film = st.text_input("🔍 Recherchez votre film :")

resultat_nom = process.extract(choix_film, df_recherche['total_title_out_KNN'].to_list(), score_cutoff = 60, limit = 20)

resultat_nom2 = []

for element in resultat_nom:
    if len(element[0]) > 4:
        resultat_nom2.append(element)


recherche = choix_film.lower().split(" ")

for element in recherche:
    df_recherche2 = df_recherche[df_recherche['total_title_out_KNN'].str.contains(element)]
    df_recherche = df_recherche2

liste_df_recherche = []

for n in range(len(df_recherche)):
    liste_df_recherche.append(df_recherche['Titre'].iloc[n])

if len(liste_df_recherche) == 0:

    liste_id_resultat_nom = []

    for element in resultat_nom2:
        id = df_recherche4[df_recherche4['total_title_out_KNN'] == element[0]]['film_id_out_KNN'].iloc[0]
        liste_id_resultat_nom.append(id)

    for element in liste_id_resultat_nom:
        nom_annee = df_recherche4[df_recherche4['film_id_out_KNN'] == element]['Titre'].iloc[0]
        liste_df_recherche.append(nom_annee)

    df_recherche = df_recherche4

if choix_film:    

    name = st.selectbox("👇 Choisissez votre film :",(liste_df_recherche))

    if name:
            
        df_selection = df[df['Titre'] == name]

        selected_film = name

        if selected_film:
            st.markdown("---")
            titre_film = selected_film
            trailer, affiche, acteur, photos, realisateur, resume = info_films(str(df_selection['film_id_out_KNN'].iloc[0]) )

            

            html_str = f"""
                <h2 class="titre_film">🎬 {df_selection['Titre'].iloc[0]}</h2>
                <p class="caract_film">{int(df_selection['year_final'])} - {str(list(df_selection['genre_out_KNN'])).replace("[", "").replace("]", "").replace('"', '').replace("'", "").capitalize()}</p> 
            """

            st.markdown(html_str, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            url_finale_title = f'{url_base_title}{str(df_selection['film_id_out_KNN'].iloc[0])}'


            html_affiche = requests.get(url_finale_title, headers={'User-Agent': navigator})
            html_affiche2 = html_affiche.content
            soup_affiche = BeautifulSoup(html_affiche2, 'html.parser')
            affiche = ''

            for balise_parent in soup_affiche.find_all('div', class_='ipc-page-content-container ipc-page-content-container--center'):
                for element in balise_parent.find_all('img', class_='ipc-image'):
                    affiche += f", {element['src']}"

            affiche = affiche.split(', ')

            if "" in affiche:
                affiche.remove("")

            lien_affiche = affiche[0]

            with col1:
                html_poster = f"""
                    <img class="img_poster" src="{lien_affiche}" />
                """

                st.markdown(html_poster, unsafe_allow_html=True)


            with col2:
                html_vote = f"""
                    <h3 class="note">⭐ Note : {round(float(df_selection['vote_exact_final']), 2)}/10</h3>
                """
                st.markdown(html_vote, unsafe_allow_html=True)

                st.html("<h3>🤵 Casting</h3>")

                html_list_actors = "<ul>"
                for a in acteur:
                    html_list_actors += f"<li>{a}</li>"
                html_list_actors += "</ul>"


                st.markdown(html_list_actors, unsafe_allow_html=True)

            with col3:

                html_resume = requests.get(url_finale_title, headers={'User-Agent': navigator})
                html_resume2 = html_resume.content
                soup_resume = BeautifulSoup(html_resume2, 'html.parser')

                for balise_parent in soup_affiche.find_all('span', class_='sc-42125d72-0 gKbnVu'):
                    resume = balise_parent.get_text().strip()


                st.html("<h3>📑 Synopsis</h3>")

                html_synopsis = f"""
                    <p>{resume}</p>
                """

                st.markdown(html_synopsis, unsafe_allow_html=True)

            st.markdown("---")

            #######################
            #
            #         KNN
            #
            #######################

            # CODE

            film_id = df_selection['film_id_out_KNN'].iloc[0]

            X = df[caracteristiques]

            df_a_predire = df[df['film_id_out_KNN'] == film_id]
            search = df_a_predire['Titre'].iloc[0]
            caracteristiques_existantes = [col for col in caracteristiques if col in df_a_predire.columns]
            df_a_predire = df_a_predire[caracteristiques_existantes]        

                    
            X_encoded, SN = encodage_X(X, 'standard', poids)

            df_final = encodage_predict(df_a_predire, SN, poids, X_encoded)

            with zipfile.ZipFile('Codes/mon_modele.pkl.zip', 'r') as zip_ref:
                with zip_ref.open('mon_modele.pkl', 'r') as f:
                    model = pickle.load(f)

            k=4

            caracteristiques.append('film_id_out_KNN')
            resultat = pokemons_similaires(df[caracteristiques], film_id, model, SN, poids, X_encoded, df)
            choix = pd.DataFrame(df[df['title_out_KNN'] == search])

            final = pd.concat([choix, resultat])
            final = final.drop(0)

            caracteristiques.remove('film_id_out_KNN')

            #######################
            #
            #       END  KNN
            #
            #######################


            st.markdown("<h2 style='text-align: center;'>🤙 Nos Rock'mendations</h2>", unsafe_allow_html=True)

            st.markdown("---")


            for n in range(8):

                genre = str(final['genre_out_KNN'].iloc[n]).replace("[", "").replace("]", "").replace('"', '').replace("'", "").capitalize()

                html_str = f"""
                    <h2 class="titre_film">🎬 {final['Titre'].iloc[n]}</h2>
                    <p class="caract_film">{int(final['year_final'].iloc[n])} - {genre}</p> 
                """

                st.markdown(html_str, unsafe_allow_html=True)

                col4, col5, col6 = st.columns(3)

                trailer, affiche, acteur, photos, realisateur, resume = info_films(str(final['film_id_out_KNN'].iloc[n]))

                with col4:
                    html_poster = f"""
                        <img class="img_poster" src="{affiche}" />
                    """

                    st.markdown(html_poster, unsafe_allow_html=True)


                with col5:
                    html_vote = f"""
                        <h3 class="note">⭐ Note : {round(float(final['vote_exact_final'].iloc[n]), 2)}/10</h3>
                    """
                    st.markdown(html_vote, unsafe_allow_html=True)

                    st.html("<h3>🤵 Casting</h3>")

                    html_list_actors = "<ul>"
                    for a in acteur:
                        html_list_actors += f"<li>{a}</li>"
                    html_list_actors += "</ul>"

                    st.markdown(html_list_actors, unsafe_allow_html=True)

                with col6:

                    url_finale_title = f'{url_base_title}{final['film_id_out_KNN'].iloc[n]}'
                    html_resume = requests.get(url_finale_title, headers={'User-Agent': navigator})
                    html_resume2 = html_resume.content
                    soup_resume = BeautifulSoup(html_resume2, 'html.parser')

                    for balise_parent in soup_resume.find_all('span', class_='sc-42125d72-0 gKbnVu'):
                        resume = balise_parent.get_text().strip()


                    st.html("<h3>📑 Synopsis</h3>")

                    html_synopsis = f"""
                        <p>{resume}</p>
                    """

                    st.markdown(html_synopsis, unsafe_allow_html=True)

                st.markdown("---")