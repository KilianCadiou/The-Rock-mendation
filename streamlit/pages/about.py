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