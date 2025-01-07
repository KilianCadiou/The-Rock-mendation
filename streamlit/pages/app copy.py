import streamlit as st
import pandas as pd
import numpy as np
import ast
import warnings
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

# Import des données

df = pd.read_csv("SRCs/P2_G5_films.csv.gz", compression="gzip")

# Import CSS

def local_css(styles):
    with open(styles) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

# --------------


st.html("<h1>The Rock'mendation</h1>")

st.html("<p>Bienvenu sur notre service de recommandation de films. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>")


# Sélectionner le film

choix_film = st.text_input("👇 Choisissez votre film")

# On vérifie si notre film existe

if choix_film:
    resultat = df[df['title_out_KNN_final'].str.contains(choix_film)]
    selected_film = st.selectbox(
        "",
        resultat['title_out_KNN_final'],
        index=None,
        placeholder="Select")
    
    # Je stock la sélection pour la similarité
    df_selection = df[df['title_out_KNN_final'] == selected_film]

    # Si mon film est sélectionné, j'affiche les suggestions 
    # dans le selecbox

    if selected_film:
        st.markdown("---")
        titre_film = selected_film


        html_str = f"""
            <h2 class="titre_film">🎬 {df_selection['title_out_KNN'].iloc[0]}</h2>
            <p class="caract_film">{int(df_selection['year_exact'])} - {str(list(df_selection['genre_out_KNN'])).replace("[", "").replace("]", "").replace('"', '').replace("'", "").capitalize()}</p> 
        """

        st.markdown(html_str, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

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
            st.image(lien_affiche, use_container_width=True)


        with col2:
            html_vote = f"""
                <h3>⭐ Note : {round(float(df_selection['vote_exact']), 2)}/10</h3>
            """
            st.markdown(html_vote, unsafe_allow_html=True)


            # On récupère nos acteurs

            html_acteurs = requests.get(url_finale_title, headers={'User-Agent': navigator})
            html_acteurs2 = html_acteurs.content
            soup_acteurs = BeautifulSoup(html_acteurs2, 'html.parser')
            liste_acteurs = []
            for balise_parent in soup_affiche.find_all('div', class_='sc-cd7dc4b7-7 vCane'):
                for element in balise_parent.find_all('a', class_='sc-cd7dc4b7-1 kVdWAO'):
                    liste_acteurs.append(element.get_text().strip())

            if len(liste_acteurs) > 4:
                liste_acteurs = liste_acteurs[:4]

            st.html("<h3>🤵 Casting</h3>")

            # st.write(liste_acteurs)

            html_list_actors = f"""
                <ul>
                    <li>{liste_acteurs[0]}</li>
                    <li>{liste_acteurs[1]}</li>
                    <li>{liste_acteurs[2]}</li>
                    <li>{liste_acteurs[3]}</li>
                </ul>
            """

            st.markdown(html_list_actors, unsafe_allow_html=True)



        # On récupère le synopsis


        html_resume = requests.get(url_finale_title, headers={'User-Agent': navigator})
        html_resume2 = html_resume.content
        soup_resume = BeautifulSoup(html_resume2, 'html.parser')

        for balise_parent in soup_affiche.find_all('span', class_='sc-3ac15c8d-1 gkeSEi'):
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


        import pandas as pd
        import numpy as np

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
                return f"Le Pokémon {film_id} n'est pas dans le dataset."

            # Récupérer les caractéristiques du Pokémon
            pokemon = X[X['film_id_out_KNN'] == film_id]

            # Je recopie ce qu'on a fait avant:
            caract_pokemon = X[X['film_id_out_KNN'] == film_id]

            caract_pokemon_encoded = encodage_predict(caract_pokemon, SN, poids, X_encoded)

            distances, indices = model.kneighbors(caract_pokemon_encoded)

            return df.iloc[indices[0]].reset_index(drop=True)


        # CODE

        #df = pd.read_csv('SRCs/P2_G5_films.csv.gz', compression = 'gzip')
        film_id = df_selection['film_id_out_KNN']

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

        X = df[caracteristiques]

        df_a_predire = df[df['film_id_out_KNN'] == film_id]
        search = df_a_predire['title_out_KNN'].iloc[0]
        # df_a_predire = df_a_predire.drop('title_len_out_KNN', axis = 1)
        # df_a_predire = df_a_predire[caracteristiques]
        caracteristiques_existantes = [col for col in caracteristiques if col in df_a_predire.columns]
        df_a_predire = df_a_predire[caracteristiques_existantes]        

        # METTRE UNIQUEMENT POUR LES COLONNES NUMERIQUES

        poids_list = pd.DataFrame(columns = caracteristiques_num, index = ['poids'])

        colonne_cle = 10
        tres_important = 7
        important = 4
        bof = 1
        rien = 0

        poids = {
            'popularity' : colonne_cle,
            'year_exact' : bof,
            'Decennie' : important,
            'runtime_exact' : rien,
            'vote_exact_tmdb' : bof,
            'arrondi_vote_exact' : colonne_cle,
            # 'vote_count_mean' : important,
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
                
        X_encoded, SN = encodage_X(X, 'standard', poids)

        df_final = encodage_predict(df_a_predire, SN, poids, X_encoded)

        with open('/Users/cedric/Desktop/Formation_DataAnalyst/The-Rockommendation/streamlit/SRCs/mon_modele.pkl', 'rb') as f: #là vous mettez l'emplacement et le nom de votre fichier pkl
            model = pickle.load(f)

        k=5

        caracteristiques.append('film_id_out_KNN')
        resultat = pokemons_similaires(df[caracteristiques], film_id, model, SN, poids, X_encoded, df)
        choix = pd.DataFrame(df[df['title_out_KNN'] == search])

        # # choix2 = choix.drop(columns = choix.columns[22:])
        # # resultat2 = resultat.drop(columns = resultat.columns[22:])

        final = pd.concat([choix, resultat])
        final = final.drop(0)

        caracteristiques.remove('film_id_out_KNN')

        st.write(final.transpose())
        st.write(final['title_out_KNN'])


        #######################
        #
        #       END  KNN
        #
        #######################


        st.html("<h2>🤙 Nos Rock'mendations</h2>")



        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image("img/gladiator.jpg", use_container_width=True)

            st.write("Titre de mon film")
            st.write("Note : 4/5")
            st.write("1990")

        with col2:
            st.image("img/gladiator.jpg", use_container_width=True)

            st.write("Titre de mon film")
            st.write("Note : 4/5")
            st.write("1990")

        with col3:
            st.image("img/gladiator.jpg", use_container_width=True)

            st.write("Titre de mon film")
            st.write("Note : 4/5")
            st.write("1990")

        with col4:
            st.image("img/gladiator.jpg", use_container_width=True)

            st.write("Titre de mon film")
            st.write("Note : 4/5")
            st.write("1990")

        with col5:
            st.image("img/gladiator.jpg", use_container_width=True)

            st.write("Titre de mon film")
            st.write("Note : 4/5")
            st.write("1990")
            



        





