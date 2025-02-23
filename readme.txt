How to run biosim :
1- git clone https://github.com/aghilhooshmand/biosim.git
2- cd biosim
3- docker-compose up --build
4- run :  localhost:8601 (it depends on port mentioned in docker compose file)
5- or for commandline run : docker-compose run biosim_cmd python data_source.csv data_target.csv data_column_pairs.csv [max_pair] [max_row_for_similarity]
- default for [max_pair] is 2 and for [max_row_for_similarity] is 5
- for calculating similarity with cbioportal change data_target with cBioPortal_clinical_features_and_values

Output is result.csv file that 
include every two sentence similarity based on model that trained by a lot of biology documents. The link of model in hugging face is :
https://huggingface.co/FremyCompany/BioLORD-2023

