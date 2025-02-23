import streamlit as st
import pandas as pd
import numpy as np
import ast
import warnings
import gzip
import zipfile
import pickle
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# modèle
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
import requests
import re
navigator = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1)'
url_base = 'https://www.imdb.com'
url_base_title = 'https://www.imdb.com/fr/title/'

import pickle


# FONCTIONS

# Fonction WebScrapping

def info_films(id):

    lien_trailer = "Aucune bande-annonce disponible"
    lien_affiche = "Aucune affiche disponible"
    liste_acteurs = []
    dico_photos_final = {}
    realisateur = "Inconnu"
    resume = "Résumé non disponible"

    url_base = 'https://www.imdb.com'
    url_base_title = 'https://www.imdb.com/fr/title/'
    url_finale_title = f'{url_base_title}{id}'

    if id == None:
        return 

    #TRAILER

    html_title = requests.get(url_finale_title, headers={'User-Agent': navigator})
    html_title2 = html_title.content
    soup_title = BeautifulSoup(html_title2, 'html.parser')

    for balise_parent in soup_title.find_all('div', class_='ipc-page-content-container ipc-page-content-container--center'):
        for element in balise_parent.find_all('a', class_='ipc-lockup-overlay ipc-focusable'):
            try:
                if 'video' in element['href']:
                    trailer = element['href']
                    lien_trailer = f'{url_base}{trailer}'
                break
            except:
                lien_trailer = "Unknown"

    

    #AFFICHE

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

    #ACTEURS

    html_acteurs = requests.get(url_finale_title, headers={'User-Agent': navigator})
    html_acteurs2 = html_acteurs.content
    soup_acteurs = BeautifulSoup(html_acteurs2, 'html.parser')
    liste_acteurs = []
    for balise_parent in soup_acteurs.find_all('div', class_='sc-cd7dc4b7-7 vCane'):
        for element in balise_parent.find_all('a', class_='sc-cd7dc4b7-1 kVdWAO'):
            liste_acteurs.append(element.get_text().strip())

    if len(liste_acteurs) > 4:
        liste_acteurs = liste_acteurs[:4]

    #PHOTOS ACTEURS

    html_acteurs = requests.get(url_finale_title, headers={'User-Agent': navigator})
    html_acteurs2 = html_acteurs.content
    soup_acteurs = BeautifulSoup(html_acteurs2, 'html.parser')
    dico_photos = {}
    dico_photos_final = {}

    for balise_parent in soup_acteurs.find_all('img', class_='ipc-image'):
        dico_photos.update({balise_parent['alt'] : balise_parent['src']})

    for element in dico_photos.keys():
        if element in liste_acteurs:
            dico_photos_final.update({element : dico_photos[element]})

    #REALISATEUR

    html_realisateurs = requests.get(url_finale_title, headers={'User-Agent': navigator})
    html_realisateurs2 = html_realisateurs.content
    soup_realisateurs = BeautifulSoup(html_realisateurs2, 'html.parser')
    liste_realisateurs = []
    for balise_parent in soup_realisateurs.find_all('a', class_='ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link'):
        liste_realisateurs.append(balise_parent.get_text().strip())

    realisateur = liste_realisateurs[0]

    #RESUME

    html_resume = requests.get(url_finale_title, headers={'User-Agent': navigator})
    html_resume2 = html_resume.content
    soup_resume = BeautifulSoup(html_resume2, 'html.parser')

    for balise_parent in soup_resume.find_all('span', class_='sc-3ac15c8d-1 gkeSEi'):
        try:
            resume = balise_parent.get_text().strip()
        except:
            resume = "Unknown"

    return lien_trailer, lien_affiche, liste_acteurs, dico_photos_final, realisateur, resume


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
    """
    Évalue différentes valeurs de k en utilisant la somme des distances aux voisins
    et le score de silhouette comme métriques.

    Args:
        X_encoded (DataFrame): Données normalisées
        k_range (range): Plage de valeurs de k à tester

    Returns:
        tuple: (distances moyennes, scores de silhouette)
    """
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans

    avg_distances = []
    silhouette_scores = []

    for k in k_range:
        # Calcul des distances moyennes pour chaque k
        from sklearn.neighbors import NearestNeighbors
        model = NearestNeighbors(n_neighbors=k)
        model.fit(X_encoded)
        distances, _ = model.kneighbors(X_encoded)
        avg_distances.append(np.mean(distances))

        # Calcul du score de silhouette
        # Nous utilisons KMeans pour créer des clusters et évaluer la qualité
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_encoded)
        if k > 1:  # Le score de silhouette nécessite au moins 2 clusters
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

    # DataFrame vide qui a les mêmes colonnes que X_encoded
    df_final = pd.DataFrame(columns=X_encoded.columns)

    # On veut que le DataFrame ait le même nombre de lignes que df_predict
    df_final = df_final.reindex(index=df_predict.index)
    # On met tous les NaN à False
    df_final = df_final.fillna(False)

    # On parcourt chaque colonne de df_predict
    # Si la colonne est présente dans X_encoded alors on la garde
    # Sinon, on la met à False
    for column in df_predict.columns:
        if column in X_encoded.columns:
            df_final[column] = df_predict[column]

    return df_final

# FONCTION 4
def pokemons_similaires(X, film_id, model, SN, poids, X_encoded, df):

    # Vérifier si le Pokémon existe dans le dataset
    if film_id not in X['film_id_out_KNN'].values:
        return f"Le film {film_id} n'est pas dans le dataset."

    # Récupérer les caractéristiques du Pokémon
    pokemon = X[X['film_id_out_KNN'] == film_id]

    # Je recopie ce qu'on a fait avant:
    caract_pokemon = X[X['film_id_out_KNN'] == film_id]

    caract_pokemon_encoded = encodage_predict(caract_pokemon, SN, poids, X_encoded)

    distances, indices = model.kneighbors(caract_pokemon_encoded)

    return df.iloc[indices[0]].reset_index(drop=True)

# Import des données

df = pd.read_csv("P2_G5_films.csv.gz", compression='gzip')

df = df.rename({'title_final_out_KNN' : 'title_out_KNN'}, axis = 1)
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

choix_film = st.text_input("🔍 Recherchez votre film")

if choix_film:

    # On vérifie si notre film existe
    df_recherche = df.copy()
    df_recherche['title_out_KNN_lower'] = df_recherche['title_out_KNN'].apply(lambda x : x.lower())
    recherche2 = choix_film.lower().split(" ")

    for element in recherche2:
        df_recherche2 = df_recherche[df_recherche['title_out_KNN_lower'].str.contains(element)]
        df_recherche = df_recherche2


    resultat = df_recherche2[df_recherche2['title_out_KNN_lower'].str.contains(choix_film)]
    
    selected_film = st.selectbox(
        "👇 Choisissez votre film",
        resultat['title_out_KNN'],
        index=None,
        placeholder="Select")
    
    



    # Je stock la sélection pour la similarité
    df_selection = resultat[resultat['title_out_KNN'] == selected_film]

    # Si mon film est sélectionné, j'affiche les suggestions 
    # dans le selecbox


    if selected_film:
        st.markdown("---")
        titre_film = selected_film
        trailer, affiche, acteur, photos, realisateur, resume = info_films(str(df_selection['film_id_out_KNN'].iloc[0]) )


        html_str = f"""
            <h2 class="titre_film">🎬 {df_selection['title_out_KNN'].iloc[0]}</h2>
            <p class="caract_film">{int(df_selection['year_final'])} - {str(list(df_selection['genre_out_KNN'])).replace("[", "").replace("]", "").replace('"', '').replace("'", "").capitalize()}</p> 
        """

        st.markdown(html_str, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        #On récupère l'affiche du film
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


            # st.image(lien_affiche, use_container_width=True)


        with col2:
            html_vote = f"""
                <h3 class="note">⭐ Note : {round(float(df_selection['vote_exact_final']), 2)}/10</h3>
            """
            st.markdown(html_vote, unsafe_allow_html=True)


            # On récupère nos acteurs

            st.html("<h3>🤵 Casting</h3>")

            # st.write(liste_acteurs)

            html_list_actors = f"""
                <ul>
                    <li>{acteur[0]}</li>
                    <li>{acteur[1]}</li>
                    <li>{acteur[2]}</li>
                    <li>{acteur[3]}</li>
                </ul>
            """

            st.markdown(html_list_actors, unsafe_allow_html=True)



        # On récupère le synopsis

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

        # # CODE

        #df = pd.read_csv('SRCs/P2_G5_films.csv.gz', compression = 'gzip')
        film_id = df_selection['film_id_out_KNN'].iloc[0]

        X = df[caracteristiques]

        df_a_predire = df[df['film_id_out_KNN'] == film_id]
        search = df_a_predire['title_out_KNN'].iloc[0]
        # df_a_predire = df_a_predire.drop('title_len_out_KNN', axis = 1)
        # df_a_predire = df_a_predire[caracteristiques]
        caracteristiques_existantes = [col for col in caracteristiques if col in df_a_predire.columns]
        df_a_predire = df_a_predire[caracteristiques_existantes]        

                
        X_encoded, SN = encodage_X(X, 'standard', poids)

        df_final = encodage_predict(df_a_predire, SN, poids, X_encoded)


        # Ouvrir le fichier ZIP
        with zipfile.ZipFile('mon_modele.pkl.zip', 'r') as zip_ref:
            # Ouvrir le fichier .pkl à l'intérieur du fichier ZIP
            with zip_ref.open('mon_modele.pkl', 'r') as f:  # Assurez-vous que le fichier s'appelle bien 'mon_modele.pkl' dans l'archive
                model = pickle.load(f)

        k=4

        caracteristiques.append('film_id_out_KNN')
        resultat = pokemons_similaires(df[caracteristiques], film_id, model, SN, poids, X_encoded, df)
        choix = pd.DataFrame(df[df['title_out_KNN'] == search])

        # # choix2 = choix.drop(columns = choix.columns[22:])
        # # resultat2 = resultat.drop(columns = resultat.columns[22:])

        final = pd.concat([choix, resultat])
        final = final.drop(0)

        caracteristiques.remove('film_id_out_KNN')

        # st.write(final['title_out_KNN'])


        #######################
        #
        #       END  KNN
        #
        #######################


        st.html("<h2>🤙 Nos Rock'mendations</h2>")


        # col1, col2, col3, col4 = st.columns(4)

        # for n in range(len(final)):
        #     if n != 0:

        #         f'trailer{n}', f'affiche{n}', f'acteur{n}', f'photos{n}', f'realisateur{n}', f'resume{n}' = info_films(final['film_id_out_KNN'].iloc[n])

        #         with f'col{n}':
        #             html_reco_title = f"""
        #                 <div class="film_reco">
        #                     <img src="{f'affiche{n}'}" />
        #                     <h3 class="titre_film_reco">{final['title_out_KNN'].iloc[n]}</h3>
        #                     <h4>⭐ Note : {round(float(final['vote_exact_final'].iloc[n]), 2)}/10</h4>
        #                     <p class="annee_film_reco">{int(final['year_final'].iloc[n])}</p>
        #                 </div>
        #             """




        trailer1, affiche1, acteur1, photos1, realisateur1, resume1 = info_films(str(final['film_id_out_KNN'].iloc[1]) )
        trailer2, affiche2, acteur2, photos2, realisateur2, resume2 = info_films(str(final['film_id_out_KNN'].iloc[2]) )
        trailer3, affiche3, acteur3, photos3, realisateur3, resume3 = info_films(str(final['film_id_out_KNN'].iloc[3]) )
        #trailer4, affiche4, acteur4, photos4, realisateur4, resume4 = info_films(str(final['film_id_out_KNN'].iloc[4]) )

        # for n in range(len(final)):
        #     f'trailer{n}', f'affiche{n}', f'acteur{n}', f'photos{n}', f'realisateur{n}', f'resume{n}' = info_films(str(final['film_id_out_KNN'].iloc[n]))

        col1, col2, col3 = st.columns(3)

        with col1:
            

            html_reco_title = f"""
                <div class="film_reco">
                    <img src="{affiche1}" />
                    <h3 class="titre_film_reco">{final['title_out_KNN'].iloc[1]}</h3>
                    <h4>Note : {round(float(final['vote_exact_final'].iloc[1]), 2)}/10</h4>
                    <p class="annee_film_reco">{int(final['year_final'].iloc[1])}</p>
                </div>
            """

            st.markdown(html_reco_title, unsafe_allow_html=True)


        with col2:
            html_reco_title = f"""
                <div class="film_reco">
                    <img src="{affiche2}" />
                    <h3 class="titre_film_reco">{final['title_out_KNN'].iloc[2]}</h3>
                    <h4>Note : {round(float(final['vote_exact_final'].iloc[2]), 2)}/10</h4>
                    <p class="annee_film_reco">{int(final['year_final'].iloc[2])}</p>
                </div>
            """

            st.markdown(html_reco_title, unsafe_allow_html=True)

        with col3:
            html_reco_title = f"""
                <div class="film_reco">
                    <img src="{affiche3}" />
                    <h3 class="titre_film_reco">{final['title_out_KNN'].iloc[3]}</h3>
                    <h4>Note : {round(float(final['vote_exact_final'].iloc[3]), 2)}/10</h4>
                    <p class="annee_film_reco">{int(final['year_final'].iloc[3])}</p>
                </div>
            """

            st.markdown(html_reco_title, unsafe_allow_html=True)

        # with col4:
        #     html_reco_title = f"""
        #         <div class="film_reco">
        #             <img src="{affiche4}" />
        #             <h3 class="titre_film_reco">{final['title_out_KNN'].iloc[4]}</h3>
        #             <h4>Note : {round(float(final['vote_exact_final'].iloc[4]), 2)}/10</h4>
        #             <p class="annee_film_reco">{int(final['year_final'].iloc[4])}</p>
        #         </div>
        #     """

            # st.markdown(html_reco_title, unsafe_allow_html=True)




        





