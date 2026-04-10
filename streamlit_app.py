import streamlit as st
import os

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

# Read the HTML file
html_file_path = os.path.join(os.path.dirname(__file__), "DRIVER DROWSINESS DETECTION.html")

try:
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Display the HTML content
    st.components.v1.html(html_content, height=800, scrolling=True)
    
except FileNotFoundError:
    st.error("HTML file not found. Please make sure 'DRIVER DROWSINESS DETECTION.html' exists in the same directory.")
    st.info("""
    ### How to fix:
    1. Make sure your HTML file is named exactly: `DRIVER DROWSINESS DETECTION.html`
    2. It should be in the same folder as this Python file
    3. Commit and push both files to GitHub
    """)