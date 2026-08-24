# auth.py (Ensure this version is used)

import streamlit as st

# --- CONFIGURATION FOR AUTHENTICATION ---
USERNAME_CORRECT = "sih_user"
PASSWORD_CORRECT = "sih_2025_pass"

def check_login(username, password, t):
    """Checks credentials and updates session state."""
    if username == USERNAME_CORRECT and password == PASSWORD_CORRECT:
        st.session_state['logged_in'] = True
        st.session_state['kyc_complete'] = False # IMPORTANT: Reset on new login
        st.session_state['pan_status'] = None    # Reset KYC feedback
        st.session_state['aadhaar_status'] = None # Reset KYC feedback
        st.rerun() # Trigger main script re-evaluation to show KYC page
    else:
        st.error(t.get("invalid_login", "Invalid Username or Password ❌"))

def logout():
    """Logs out the user and reruns the app."""
    st.session_state['logged_in'] = False
    st.session_state['kyc_complete'] = False # Clean up state
    st.rerun()

def login_page(t):
    # ... (Your existing login_page function remains the same)
    st.set_page_config(page_title=t["login_title"], layout="centered")
    st.title(t["login_title"])
    st.markdown("---")

    with st.form("login_form"):
        username = st.text_input(t["username"])
        password = st.text_input(t["password"], type="password")
        submitted = st.form_submit_button(t["login_button"])
        if submitted:
            check_login(username, password, t)
            # The check_login handles the rerun.