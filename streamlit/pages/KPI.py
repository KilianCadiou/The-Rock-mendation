import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import CSS --------

def local_css(styles):
    with open(styles) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

# --------------------

st.html("<h2>📊 Quelques chiffres clés</h2>")


# KPI 1


st.markdown("""

    ## 📊 KPI 1 : Nombre de films par acteur/actrice
            
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
            
        """)

# Lecture du fichier
df = pd.read_csv('../kpi_extracts/KPI1Extract.csv.gz')
df = df.dropna(subset=['Acteur/Actrice'])
df['Acteur/Actrice'] = df['Acteur/Actrice'].astype(str)

# On affiche les 25 premiers
df_test = df.head(25)

# Affichage du graphique
fig = plt.figure(figsize=(10, 6))
plt.plot(df_test['Acteur/Actrice'], df_test['Nombre de films'], label='KPI1', marker='o')

plt.xlabel('Acteur/Actrice')
plt.ylabel('Nombre de films')
# plt.title('Nombre de films par acteur/actrice')
plt.xticks(rotation=90)
plt.legend()

# plt.tight_layout()
st.pyplot(fig)



# KPI2

st.markdown("""

    ## 📊 KPI 2 : Durée des films par années
            
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
            
            """)

# Lecture du fichier
df = pd.read_csv('../kpi_extracts/KPI2Extract.csv.gz')

# Moyenne des durées de films
df_test = df.groupby('year_exact')['runtime_exact'].mean()

# Affichage du graph
fig2 = plt.figure(figsize=(10, 6))
plt.plot(df_test, label='KPI2', marker='o')

plt.xlabel('Années des films')
plt.ylabel('Durée des films')
# plt.title('Durée des films par années')
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
st.pyplot(fig2)


# KPI 3

st.markdown("""

    ## 📊 KPI 3 : Nombres d'acteur en fonction de leurs âges au moment du film (ou série)
            
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
            
            """)

# Lecture des fichiers

df = pd.read_csv('../kpi_extracts/KPI3ActeurTV.csv.gz',)
df_duo = pd.read_csv('../kpi_extracts/KPI3Duo.csv.gz',)
df_film = pd.read_csv('../kpi_extracts/KPI3Film.csv.gz')

# Calcul des âges
df_total = df.groupby('birthYear')['nconst'].count()
test_duo = df_duo.groupby('birthYear')['nconst'].count()
test_film = df_film.groupby('birthYear')['nconst'].count()
test_film = test_film[test_film.index>1800]
test_duo = test_duo[test_duo.index>1800]

# Affichage du graph
fig3 = plt.figure(figsize=(10, 6))
plt.plot(test_film, label='Film', marker='o')
plt.plot(df_total, label='Serie', marker='o')
plt.plot(test_duo, label='Duo', marker='o')

plt.xlabel('Année de naissance des acteurs')
plt.ylabel("Nombre d'acteur en fonction des carrières")
# plt.title("Nombres d'acteur en fonction des années de naissance par film ou série")
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
st.pyplot(fig3)



# KPI 4

st.markdown("""

    ## 📊 KPI 4 : Âge des acteurs au moment des tournages
            
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
            
            """)

df = pd.read_csv('../kpi_extracts/KPI4Bis.csv.gz',)


test_film = df.groupby('age')['nconst'].count()
test_film = test_film[test_film.index>10]
test_film = test_film[test_film.index<100]
test2 = pd.DataFrame(test_film).reset_index()

test2['Tranches'] = ''
test2['age'] = test2['age'].fillna('0')
test2['age'] = test2['age'].replace('nan', '0')
test2['age'] = test2['age'].astype(str)
test2['Tranches'] = test2['age'].str[:1] + "0"
test2['age'] = pd.to_numeric(test2['age'])
test2['Tranches'] = pd.to_numeric(test2['Tranches'])

test3 = pd.DataFrame(test2['Tranches'].value_counts()).reset_index()



fig4 = plt.figure(figsize=(8, 8))
plt.pie(test3['Tranches'], labels = test3['Tranches'], autopct='%1.1f%%', startangle=90)
# plt.title('Âge des acteurs au moment des tournages')
st.pyplot(fig4)


# KPI 5

st.markdown("""

    ## 📊 KPI 5 : Budget, période et genre des films les mieux notés
            
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
            
            """)

# Lecture des fichiers
budget = pd.read_csv('../kpi_extracts/KPI5Budget.csv.gz',)
Decennie = pd.read_csv('../kpi_extracts/KPI5Decennie.csv.gz',)
Genre = pd.read_csv('../kpi_extracts/KPI5Genre.csv.gz',)


st.markdown("""

    ### Budget des meilleurs films
            
            """)

# Affichage du premier graph
fig5a = plt.figure(figsize=(8, 8))
plt.pie(budget['Apparition'], labels = budget['Budget'], autopct='%1.1f%%', startangle=90)
plt.title('Budget des meilleurs films')
st.pyplot(fig5a)



st.markdown("""

    ### Les périodes des films les mieux notés
            
            """)

# Création d'un DF optimisé
data = {
    'decennie': ['Avant 1950', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s'],
    'count': [6, 2803, 4390, 5366, 6545, 8490, 18738, 37770, 20569],
    'Apparition': [8.98, 2.56, 4.03, 4.93, 5.99, 7.79, 17.17, 34.63, 18.86]
}

df = pd.DataFrame(data)

# Affichage du graph
fig5b = plt.figure(figsize=(8, 8))
plt.pie(df['Apparition'], labels=df['decennie'], autopct='%1.1f%%', startangle=90)
# plt.title('Les périodes des films les mieux notés')
st.pyplot(fig5b)


st.markdown("""

    ### Les genres des films les mieux notés
            
            """)

# Création d'un DF optimisé
data2 = {
    'Genre': ['Autres', 'Family', 'Adventure', 'Biography', 'Thriller', 'History', 'Action', 'Crime', 'Music', 'Romance', 'Comedy', 'Documentary', 'Drama'],
    'count': [8370, 2356, 2493, 2851, 2873, 3076, 3282, 3404, 3469, 5214, 8155, 13576, 19877],
    'Pourcentage apparition': [11.97, 3.37, 3.57, 4.08, 4.11, 4.41, 4.71, 4.89, 5.00, 7.51, 11.77, 19.58, 28.02] 
}

df = pd.DataFrame(data2)

# Affichage du graph
fig5c = plt.figure(figsize=(8, 8))
plt.pie(df['count'], labels=df['Genre'], autopct='%1.1f%%', startangle=90)
# plt.title('Les genres des films les mieux notés')
st.pyplot(fig5c)