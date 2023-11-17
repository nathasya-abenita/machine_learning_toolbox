import streamlit as st

st.set_page_config(
    page_title="Basic Machine Learning Tool",
    page_icon="👋"
)

st.title("Welcome to Basic Machine Learning Tool! 👋")

st.sidebar.success("Select a tool above.")

# %% Introduction
st.info(
"""
There are three tools available. Please choose a page to use the desired tool.
""", icon = 'ℹ️'
)