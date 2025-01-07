import streamlit as st

# Import CSS

def local_css(styles):
    with open(styles) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

st.html(
   " <ul>"
        "<li>Structure page App ⌛</li>"
        "<li>Créer logo du service ⌛</li>"
        "<li>Bakground App ✅</li>"
        "<li>Menu gauche + emoji ✅</li>"
        "<li>Vérifier les typos ✅</li>"
    "</ul>"
)