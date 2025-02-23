import streamlit as st
import pandas as pd
from BioSimilarity import get_similarity

# Title and Description
st.title("BioSimilarity AI – Intelligent Feature Comparison for Biology")

st.markdown("""
BioSimilarity AI is an advanced tool that calculates feature similarity using a specialized AI model trained on biological data. 
Designed for researchers and data scientists, it enables precise comparison of datasets, aiding in biomarker discovery, clinical studies, and biological data analysis. 🚀
""")

# Function to reset the app
def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Sidebar Controls
with st.sidebar:
    # Read Me button (shows info)
    if st.button("📖 Read Me", key="read_me_button"):
        st.markdown(
            "### About This App\n"
            "- This tool allows you to compare biosimilarity between datasets.\n"
            "- Upload two CSV files and select columns to compare.\n"
            "- Click 'Add Pair' to save column selections.\n"
            "- Use 'Clear Pairs' to reset selections or 'Clear All' to restart the app.\n"
            "- [📂 Open Read Me](readme.txt) to view full documentation."
        )

    # Link to cBioPortal
    st.markdown("[🔗 Visit cBioPortal](https://www.cbioportal.org)", unsafe_allow_html=True)

    # Reset button
    if st.button("🔄 Clear All (Reset App)", key="sidebar_clear_all"):
        reset_app()

# File Upload Section (First Step)
with st.expander("📂 Upload CSV Files", expanded=True):
    uploaded_file1 = st.file_uploader("Choose the first CSV file", type="csv")

    # Option to use cBioPortal features instead of manually uploading
    use_cbioportal_features = st.checkbox("Use cBioPortal Features file instead of uploading", key="cbioportal_checkbox")

    # Second file uploader (only shown if the checkbox is NOT checked)
    uploaded_file2 = None
    if not use_cbioportal_features:
        uploaded_file2 = st.file_uploader("Choose the second CSV file", type="csv")

# Only show additional sections if a first file is uploaded and either a second file is uploaded OR the checkbox is checked
if uploaded_file1 and (uploaded_file2 or use_cbioportal_features):
    # Data Preview Section
    with st.expander("🔍 Data Preview", expanded=True):
        df1 = pd.read_csv(uploaded_file1, delimiter='\t', on_bad_lines='skip')

        # Load the second dataset based on user choice
        if use_cbioportal_features:
            df2 = pd.read_csv("cBioPortal_clinical_features_and_values.csv",delimiter='\t', on_bad_lines='skip')
            st.success("Loaded cBioPortal_clinical_features_and_values.csv successfully!")
        else:
            df2 = pd.read_csv(uploaded_file2, delimiter='\t', on_bad_lines='skip')

        st.write("First CSV file data:")
        st.write(df1)
        st.write("Second CSV file data:")
        st.write(df2)

    # Feature Selection Section
    with st.expander("📂 Select Features"):
        col1 = st.selectbox("Select a column from the first CSV file", df1.columns)
        col2 = st.selectbox("Select a column from the second CSV file", df2.columns)

        # Initialize session state for column pairs
        if "data_column_pairs" not in st.session_state:
            st.session_state.data_column_pairs = pd.DataFrame(columns=["Pairs"])

        # Button to add column pairs
        if st.button("➕ Add Pair", key="add_pair_button"):
            new_row = {"Pairs": col1 + ";" + col2}
            st.session_state.data_column_pairs = pd.concat(
                [st.session_state.data_column_pairs, pd.DataFrame([new_row])], ignore_index=True
            )

        # Display selected column pairs
        st.write("### Selected Column Pairs:")
        st.write(st.session_state.data_column_pairs)

        # Button to clear only column pairs
        if st.button("🗑️ Clear Pairs", key="clear_pairs_button"):
            st.session_state.data_column_pairs = pd.DataFrame(columns=["Pairs"])
            st.rerun()

# Similarity Calculation Section
with st.expander("🔬 Calculate Similarity", expanded=True):
    if st.button("⚙️ Calculate Similarity", key="calculate_similarity_button"):
        if not st.session_state.data_column_pairs.empty:
            # Calculate similarity
            similarity_results = get_similarity(df1, df2, st.session_state.data_column_pairs,2,5)

            # Display similarity results
            st.write("### Similarity Results:")
            st.write(similarity_results)
        else:
            st.warning("No column pairs selected. Please add column pairs to calculate similarity.")


