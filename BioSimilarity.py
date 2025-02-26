import pandas as pd
import numpy as np
from sentence_transformers import  util
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import sys,os,tqdm
from tqdm.auto import tqdm


#Append two columns that are going to be compared
def choose_columns_for_calculate_similarity(df_pairs: pd.DataFrame, df1_col: str, df2_col: str) -> pd.DataFrame:
    concatenated_value = f"{df1_col};{df2_col}"
    new_row = pd.DataFrame([{"column_pair": concatenated_value}])  # Use a proper column name
    df_pairs = pd.concat([df_pairs, new_row], ignore_index=True)
    return df_pairs

#Calculate similarity between two columns
def sentence_similarity_by_torch_BioLORD(s1: list, s2: list, max_number_similarity: int):
    
    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained('FremyCompany/BioLORD-STAMB2-v1', cache_dir="bioSim_model/")
    model = AutoModel.from_pretrained('FremyCompany/BioLORD-STAMB2-v1', cache_dir="bioSim_model/").to(device)

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    result = pd.DataFrame(columns=['s1', 's2', 'similarity'])
    count = 0
    total_calculations = len(s1) * len(s2)
    
    with tqdm(total=total_calculations, desc="Calculating similarities") as pbar:
        for i in s1:
            for j in s2:
                if not i or not j:
                    result.loc[len(result)] = [i, j, None]
                    pbar.update(1)
                    continue
                try:
                    sentences = [i, j]
                    if i == " " or j == " ":
                        result.loc[len(result)] = [i, j, 0]
                    else:
                        # Tokenize sentences
                        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
                        # Compute token embeddings
                        with torch.no_grad():
                            model_output = model(**encoded_input)
                        # Perform pooling
                        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                        # Normalize embeddings
                        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                        similarity = util.cos_sim(sentence_embeddings[0], sentence_embeddings[1]).cpu().data.numpy()[0][0]
                        
                        result.loc[len(result)] = [i, j, np.round(similarity, 2)]
                    count += 1
                    pbar.update(1)
                    if count == max_number_similarity:
                        return result
                except Exception as e:
                    print(f"Error processing pair ({i}, {j}): {e}")
                    result.loc[len(result)] = [i, j, None]
                    pbar.update(1)
    return result
     
    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained('FremyCompany/BioLORD-STAMB2-v1', cache_dir="bioSim_model/")
    model = AutoModel.from_pretrained('FremyCompany/BioLORD-STAMB2-v1', cache_dir="bioSim_model/").to(device)

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    result = pd.DataFrame(columns=['s1', 's2', 'similarity'])
    count=0
    for i in s1:
        for j in s2:
            if not i or not j:
                result.loc[len(result)] = [i, j, None]
                continue
            try:
                sentences = [i, j]
                if i==" " or j==" " :
                    result.loc[len(result)] = [i, j, 0]
                else :    
                    # Tokenize sentences
                    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
                    # Compute token embeddings
                    with torch.no_grad():
                        model_output = model(**encoded_input)
                    # Perform pooling
                    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                    # Normalize embeddings
                    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                    similarity = util.cos_sim(sentence_embeddings[0], sentence_embeddings[1]).cpu().data.numpy()[0][0]
                    
                    result.loc[len(result)] = [i, j, np.round(similarity, 2)]
                count+=1
                calculations_done+=1
                print(f"Calculations done: {calculations_done}/{all_calculations} ({(calculations_done/all_calculations)*100:.2f}% completed)")
                if count==max_number_similarity:
                    return result,calculations_done
            except Exception as e:
                print(f"Error processing pair ({i}, {j}): {e}")
                result.loc[len(result)] = [i, j, None]
    return result,calculations_done



def get_similarity(data_source: pd.DataFrame, data_target: pd.DataFrame, column_pairs: pd.DataFrame,max_number_pair:int=2,max_number_similarity:int=5) -> pd.DataFrame:
    All_similarity = pd.DataFrame()
    count=0
    for i in column_pairs.index:
        print("...........................................")     
        print(f"Processing pair {i+1}/{max_number_pair} ...")
        print("...........................................")
        col_x, col_y = column_pairs.iloc[i].values[0].split(";")
        s1 = data_source[col_x].fillna(" ").tolist()
        s2 = data_target[col_y].fillna(" ").tolist()
        similarity_df = sentence_similarity_by_torch_BioLORD(s1, s2,max_number_similarity)
        similarity_df.columns = [f"{col_x}_{i}" , f"{col_y}_{i}" ,f"Similarity( {col_x} - {col_y} )"]
        All_similarity = pd.concat([All_similarity, similarity_df], axis=1)
        count+=1
        if count==max_number_pair:
            return All_similarity
    return All_similarity


def main():
    current_working_directory = os.getcwd()
    if len(sys.argv) < 4:
        print("Usage: python BioSimilarity.py <source_file> <target_file> <column_pairs_file> <max_number_pair> <max_number_similarity>")
        sys.exit(1)
    else:
        data_source = pd.read_csv(current_working_directory + '/' + sys.argv[1], delimiter='\t')
        data_target = pd.read_csv(current_working_directory + '/' + sys.argv[2], delimiter='\t')
        data_column_pairs = pd.read_csv(current_working_directory + '/' + sys.argv[3], delimiter='\t')
        if len(sys.argv) > 4 and sys.argv[4]:
            max_number_pair = int(sys.argv[4])
        else:
            max_number_pair = len(data_column_pairs)
        if len(sys.argv) > 5 and sys.argv[5]:
            max_number_similarity = int(sys.argv[5])
        else:
            max_number_similarity = 0
            for i in data_column_pairs.index:
                 max_number_similarity=max_number_similarity+len(data_source)*len(data_target)    
        result = get_similarity(data_source, data_target, data_column_pairs,max_number_pair,max_number_similarity)
        result.to_csv(current_working_directory + '/' + 'result.csv', sep='\t', index=False)
    

if __name__ == '__main__':
    main()
