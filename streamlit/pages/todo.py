import streamlit as st

# Import CSS

def local_css(styles):
    with open(styles) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")


st.html(
    "<h2>A faire</h2>"
   " <ul>"
        "<li>Structure page App ✅</li>"
        "<li>Créer logo du service ✅</li>"
        "<li>Bakground App ✅</li>"
        "<li>Menu gauche + emoji ✅</li>"
        "<li>Vérifier les typos ✅</li>"
        "<li>Page About ✅</li>"
        "<li>Photo team ✅</li>"
        "<li>Reload graph express ⌛</li>"
        "<li>st.cache_ressources ⌛</li>"

    "</ul>"
)


st.html(
    "<h2>Piste d'évolution</h2>"
    " <ul>"
        "<li>Possibilité de mettre en favoris ses films à voir</li>"
        "<li>Ajout de différents filtres de recherche</li>"
        "<li>Ajout d'une page contact</li>"
    "</ul>"
)
