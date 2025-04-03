import streamlit as st
import pandas as pd
import plotly.express as px
import tarfile

# Function to load data
@st.cache_data
def load_data():
    def extract_tar_xz(file_path, output_dir):
        """Extracts a .tar.xz archive to the specified output directory."""
        try:
            with tarfile.open(file_path, "r:xz") as tar:
                tar.extractall(path=output_dir)
                print(f"Extracted '{file_path}' to '{output_dir}' successfully.")
        except Exception as e:
            print(f"Error extracting '{file_path}': {e}")

    # Example Usage
    extract_tar_xz("cBioPortal_clinical_features_and_values_all_data_similarity_Breast.tar.xz", ".")
    
    return pd.read_csv('cBioPortal_clinical_features_and_values_all_data_similarity_Breast.csv', 
                       on_bad_lines='skip', delimiter='\t')

# Function to filter data based on sidebar selections
def filter_data(data):
    st.sidebar.header('Filter Options')
    clinical_features = st.sidebar.multiselect('Clinical Features', data['clinical_feature_0'].unique())
    cancer_colors = st.sidebar.multiselect('Cancer Colors', data['cancer_color'].unique())
    cancer_types = st.sidebar.multiselect('Cancer Types', data['cancer_type'].unique())
    data_types = st.sidebar.multiselect('Data Types', data['data_type'].unique())
    patient_samples = st.sidebar.multiselect('Patient or Sample', data['patient_sample'].unique())

    # Use all values if none selected
    clinical_features = clinical_features or data['clinical_feature_0'].unique()
    cancer_colors = cancer_colors or data['cancer_color'].unique()
    cancer_types = cancer_types or data['cancer_type'].unique()
    data_types = data_types or data['data_type'].unique()
    patient_samples = patient_samples or data['patient_sample'].unique()

    # Similarity sliders
    st.sidebar.header('Similarity Measures')
    similarity_cf_cf = st.sidebar.slider('Clinical Feature - Clinical Feature', 0.0, 1.0, (0.0, 1.0))
    similarity_cf_desc = st.sidebar.slider('Clinical Feature - Description', 0.0, 1.0, (0.0, 1.0))
    similarity_desc_desc = st.sidebar.slider('Description - Description', 0.0, 1.0, (0.0, 1.0))
    similarity_default = st.sidebar.slider('Default Value - Default Value', 0.0, 1.0, (0.0, 1.0))
    average_similarity = st.sidebar.slider('Average Similarity', 0.0, 1.0, (0.0, 1.0))

    # Apply filtering
    return data[
        (data['clinical_feature_0'].isin(clinical_features)) &
        (data['cancer_color'].isin(cancer_colors)) &
        (data['cancer_type'].isin(cancer_types)) &
        (data['data_type'].isin(data_types)) &
        (data['patient_sample'].isin(patient_samples)) &
        (data['Similarity( clinical_feature - clinical_feature )'].between(*similarity_cf_cf)) &
        (data['Similarity( clinical_feature - description )'].between(*similarity_cf_desc)) &
        (data['Similarity( description - description )'].between(*similarity_desc_desc)) &
        (data['Similarity( Default_value - Default_value )'].between(*similarity_default)) &
        (data['Average'].between(*average_similarity))
    ]

# Function to display an interactive table with column filters
def display_table(data):
    st.write("### Filtered Data Table")
    st.data_editor(data, use_container_width=True, hide_index=True)

# Function to create and display a heatmap
def display_heatmap(data):
    st.write("### Similarity Heatmap")
    heatmap_data = data.pivot(index='clinical_feature_1', columns='clinical_feature_0', values='Average')

    fig = px.imshow(
        heatmap_data.values, 
        labels={"x": "Clinical Feature 0", "y": "Clinical Feature 1", "color": "Average"},
        x=heatmap_data.columns, 
        y=heatmap_data.index, 
        color_continuous_scale='Viridis'
    )

    st.plotly_chart(fig, use_container_width=True)

# Function to create and display a scatter plot
def display_scatter_plot(data):
    st.write("### Matrix Grid Chart (Scatter Plot)")
    fig = px.scatter(
        data, 
        x='clinical_feature_0', 
        y='clinical_feature_1', 
        size='Average', 
        color='Average',
        color_continuous_scale='Viridis'
    )

    st.plotly_chart(fig, use_container_width=True)

# Main function
def main():
    st.title('Similarity Map Explorer')
    st.write('Use the sidebar to filter the data and explore the similarity heatmap.')

    # Load and filter data
    data = load_data()
    filtered_data = filter_data(data)

    # Display components
    display_table(filtered_data)
    #display_heatmap(filtered_data)
    display_scatter_plot(filtered_data)

# Run app
if __name__ == '__main__':
    main()
