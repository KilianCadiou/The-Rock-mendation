import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

# Import CSS

def local_css(styles):
    with open(styles) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("streamlit/styles.css")

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
        "<li>Réaliser une <a href='https://docs.google.com/document/d/11CvkiZSQv0-sk87al2F1MfSFFesAjdreX2VEsOvnPac/edit?tab=t.0' target='_blank'>étude de marché</a> sur la consommation de cinéma dans la région.</li>"
        "<li>Mettre en avant certains <a href='KPI'>chiffres clés (KPI)</a> comme les acteurs les plus présents, l'âge moyen des acteurs...</li>"
        "<li>Créer une <a href='app'>Application</a> de recommandation de film en fonction des appréciations du spectateur.</li>"
    "</ol>"
)


st.html(
    "<h2>⚙️ Stack technique : </h2>"
)

col1, col2, col3, col4 = st.columns(4)

with col1: 
    st.image("streamlit/img/Python.png")

with col2:
    st.image("streamlit/img/pandas_white.png")

with col3:
    st.image("streamlit/img/scikit-learn.png")

with col4:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-lighttext.png")


import streamlit as st

# Import CSS

def local_css(styles):
    with open(styles) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("streamlit/styles.css")


st.html("<h1 id='a-propos'>🙋‍♂️ A propos</h1>")

st.html(
    "<p>Nous sommes une équipe de passionnés de données et de cinéma, réunis par un projet ambitieux : concevoir un système de recommandation de films dédié à la Creuse. Combinant nos compétences en analyse de données, en machine learning et en visualisation, nous avons travaillé main dans la main pour créer une solution qui inspire et divertit. Chacun de nous apporte une expertise unique, qu’il s’agisse de programmation, de gestion de projet ou encore de créativité dans l’approche des problématiques. Ce projet est le fruit de notre collaboration, de nos échanges d’idées et de notre envie commune de transformer les données en une expérience accessible et personnalisée.</p>"
)

st.html("<h2>👨🏻‍💼 La team</h2>")
            
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image("streamlit/img/avatar-malo.png")
    st.html(    
        "<h4>Malo L.</h4>"
        "<p><a href='https://www.linkedin.com/in/malo-le-pors-5373a8273/' target='_blank'>LinkedIn</a></p>"
        "<p><a href='https://github.com/MaloBang' target='_blank'>Github</a></p>"
    )

with col2:
    st.image("streamlit/img/avatar-kilian.png")
    st.html(    
        "<h4>Kilian C.</h4>"
        "<p><a href='https://www.linkedin.com/in/kiliancadiou/' target='_blank'>LinkedIn</a></p>"
        "<p><a href='https://github.com/KilianCadiou' target='_blank'>Github</a></p>"
    )

with col3:
    st.image("streamlit/img/avatar-romain.png")
    st.html(    
        "<h4>Romain F.</h4>"
        "<p><a href='https://www.linkedin.com/in/romain-foucault-01b11a15a/' target='_blank'>LinkedIn</a></p>"
        "<p><a href='https://github.com/LegacyLord44' target='_blank'>Github</a></p>"
    )

with col4:
    st.image("streamlit/img/avatar-cedric.png")
    st.html(    
        "<h4>Cédric R.</h4>"
        "<p><a href='https://www.linkedin.com/in/c3dr1c/' target='_blank'>LinkedIn</a></p>"
        "<p><a href='https://github.com/DriixData' target='_blank'>Github</a></p>"
    )
    