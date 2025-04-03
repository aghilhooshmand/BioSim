
# How to Run BioSim

### 1. Clone the Repository  
```
git clone https://github.com/aghilhooshmand/biosim.git
```

### 2. Navigate to the Project Directory  
```
cd biosim
```

### 3. Build and Run Using Docker  
```
docker-compose up --build
```

### 4. Access the Application  
- Open your browser and go to:  
  ```
  http://localhost:8601
  ```
  *(Port number depends on the configuration in the `docker-compose` file.)*

### 5. Command-Line Execution  
To run the application via the command line, use:  
```
docker-compose run biosim_cmd python data_source.csv data_target.csv data_column_pairs.csv [max_pair] [max_row_for_similarity]
```
- If `[max_pair]` and `[max_row_for_similarity]` are not provided, the program will process all data.

### 6. Similarity Calculation with cBioPortal  
- Instead of uploading `data_target.csv`, you can select the **"cBioPortal_clinical_features_and_values"** checkbox for similarity calculations.

### Output  
- The result will be saved as `result.csv`, which can be filtered by columns and downloaded.  
- Each row includes sentence similarity scores based on a model trained on extensive biological documents.

### Model Information  
The model used for similarity calculations is trained on a vast corpus of biological texts. You can find it on Hugging Face:  
🔗 **[BioLORD-2023](https://huggingface.co/FremyCompany/BioLORD-2023)**  
