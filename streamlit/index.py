import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

# Import CSS

def local_css(styles):
    with open(styles) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

st.html("<h1>The Rock'mendation <br> <span>Le projet</span></h1>")
st.markdown("---")

st.html(
    "<p>Notre équipe de Data Analyst a été mandaté par un cinéma dans la Creuse qui se trouve en difficulté voulant passer le cap du digital.</p>"
)

st.html(
    "<h2>🚀 Objectifs et enjeux :</h2>"
)

st.html(

    "<ol class='liste-objectifs'>"
        "<li>Réaliser une <a href='https://google.fr' target='_blank'>étude de marché</a> sur la consommation de cinéma dans la région.</li>"
        "<li>Mettre en avant certains <a href='KPI'>chiffres clés (KPI)</a> comme les acteurs les plus présents, l'âge moyen des acteurs...</li>"
        "<li>Créer une <a href='app'>Application</a> de recommandation de film en fonction des appréciations du spectateur.</li>"
    "</ol>"
)


st.html(
    "<h2>⚙️ Stack technique : </h2>"
)

col1, col2, col3, col4 = st.columns(4)

with col1: 
    st.image("img/python.png")

with col2:
    st.image("img/pandas_white.png")

with col3:
    st.image("img/scikit-learn.png")

with col4:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-lighttext.png")
