"""Streamlit Community Cloud entry point: streamlit run streamlit_app.py."""

import streamlit as st

# Root-level Cloud secrets become environment variables used by api services.
# Local runs may instead supply their environment through the shell.
try:
    st.secrets.to_dict()
except FileNotFoundError:
    pass

from ui.app import main

main()
