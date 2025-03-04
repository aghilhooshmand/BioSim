import pandas as pd


# Load the data
all_data_similarity = pd.read_csv('result.csv',on_bad_lines='skip',delimiter='\t')
all_data = pd.read_csv('cBioPortal_clinical_features_and_values.csv',on_bad_lines='skip',delimiter='\t')


# Merge the data on 'clinical feature' and 'clinical feature_0'
merged_data = pd.merge(all_data_similarity, all_data, 
                       left_on='clinical_feature_0', right_on='clinical_feature', 
                       how='left')

# Add cancer color, cancer type, patient or sample, and data type to file1
all_data_similarity['cancer_color'] = merged_data['cancer color']
all_data_similarity['cancer_type'] = merged_data['cancer type']
all_data_similarity['patient_or_sample'] = merged_data['patient or sample']
all_data_similarity['data_type'] = merged_data['data type']


# Save the updated file1
all_data_similarity.to_csv('cBioPortal_clinical_features_and_values_all_data_similarity.csv', index=False,sep='\t')

