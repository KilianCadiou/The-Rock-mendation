import streamlit as st
from st_pages import get_nav_from_toml

nav = get_nav_from_toml("streamlit/.streamlit/pages.toml")

st.logo("streamlit/img/logo.png")



pg = st.navigation(nav)

# add_page_title(pg)

pg.run()