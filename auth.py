
# auth.py

import streamlit as st

# --- CONFIGURATION FOR AUTHENTICATION ---
USERNAME_CORRECT = "sih_user"
PASSWORD_CORRECT = "sih_2025_pass"


def check_login(username, password, t):
    """Checks credentials and updates session state."""
    if username == USERNAME_CORRECT and password == PASSWORD_CORRECT:
        st.session_state['logged_in'] = True
        st.session_state['kyc_complete'] = False  # Reset on new login
        st.session_state['pan_status'] = None
        st.session_state['aadhaar_status'] = None

        st.rerun()
    else:
        st.error(t.get("invalid_login", "Invalid Username or Password ❌"))


def logout():
    """Logs out the user and reruns the app."""
    st.session_state['logged_in'] = False
    st.session_state['kyc_complete'] = False
    st.rerun()


def login_page(t):
    """Displays the login page."""

    st.title(t["login_title"])
    st.markdown("---")

    with st.form("login_form"):
        username = st.text_input(t["username"])
        password = st.text_input(
            t["password"],
            type="password"
        )

        submitted = st.form_submit_button(t["login_button"])

        if submitted:
            check_login(username, password, t)
