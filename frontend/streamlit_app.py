# frontend/streamlit_app.py
import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Document Intelligence", layout="wide")

st.title("📄 Document Intelligence System")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Document Processing", "Search", "Classification"])

if page == "Document Processing":
    st.header("Process Documents")
    
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['pdf', 'png', 'jpg', 'jpeg']
    )
    
    if uploaded_file:
        with st.spinner("Processing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_URL}/process-document", files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Document Info")
                    st.write(f"**Filename:** {result['filename']}")
                    st.write(f"**Document ID:** {result['doc_id']}")
                    
                    st.subheader("Summary")
                    st.write(result['summary'])
                
                with col2:
                    st.subheader("Entities Found")
                    entities_df = pd.DataFrame(result['entities'])
                    st.dataframe(entities_df)
                
                st.subheader("Text Preview")
                st.text(result['text_preview'])
            else:
                st.error(f"Error: {response.text}")

elif page == "Search":
    st.header("Search Documents")
    
    query = st.text_input("Enter your search query")
    n_results = st.slider("Number of results", 1, 10, 5)
    
    if st.button("Search") and query:
        with st.spinner("Searching..."):
            response = requests.post(
                f"{API_URL}/search",
                json={"query": query, "n_results": n_results}
            )
            
            if response.status_code == 200:
                results = response.json()['results']
                
                for i, doc in enumerate(results['documents'][0]):
                    st.subheader(f"Result {i+1}")
                    st.write(doc)
                    st.write(f"**Score:** {results['distances'][0][i]:.3f}")
                    st.divider()
            else:
                st.error(f"Error: {response.text}")

elif page == "Classification":
    st.header("Text Classification")
    
    text = st.text_area("Enter text to classify")
    labels = st.text_input("Enter labels (comma-separated)", "urgent, normal, spam")
    
    if st.button("Classify") and text:
        label_list = [l.strip() for l in labels.split(",")]
        
        with st.spinner("Classifying..."):
            response = requests.post(
                f"{API_URL}/classify",
                json={"text": text, "labels": label_list}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                st.subheader("Classification Results")
                for label, score in zip(result['labels'], result['scores']):
                    st.progress(score)
                    st.write(f"{label}: {score:.2%}")
            else:
                st.error(f"Error: {response.text}")