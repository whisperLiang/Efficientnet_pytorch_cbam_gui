import pandas as pd
import os
import json

def create_map(csv_filename, txt_name):
    # if not os.path.exists(txt_name):
        # os.makedirs(txt_name)
    data_file = pd.read_csv(csv_filename)
    id_list = data_file["ID"].values.tolist()
    # classes_list = data_file["ScientificName"].values
    classes_list = data_file["SpeciesID"].values.tolist()
    dict_map = dict(zip(id_list, classes_list))
    # dict_map = dict(zip(id_list, id_list))
    json_map = json.dumps(dict_map)
    # print(json_map)
    with open(txt_name, 'w', encoding='utf8') as f:
        f.write(json_map)

if __name__ == "__main__":
    # csv_filename = "af2020cv-2020-05-09-v5-dev/species.csv"
    # txt_name = "underwater.txt"
    csv_filename = "./gesture.csv"
    txt_name = "gesture.txt"
    create_map(csv_filename, txt_name)
    
    
