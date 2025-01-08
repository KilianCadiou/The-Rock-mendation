import streamlit as st
import base64
from st_pages import add_page_title, get_nav_from_toml

nav = get_nav_from_toml(
    ".streamlit/pages.toml"
)

st.logo("img/logo.png")


pg = st.navigation(nav)

# add_page_title(pg)

pg.run()