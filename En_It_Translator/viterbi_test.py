import os.path

import pandas as pd
import json

from pos_tagger import compute_emission_matrix

if __name__ == "__main__":

    file_name = './data/ud-word-pos-ewt-test.csv'
    test_corpus_df = pd.read_csv(file_name, sep='\t')
    obs = " ".join(test_corpus_df['word'].tolist())

    if os.path.exists('./data/emission_matrix_test.json'):
        print("loading emission matrix...")
        with open('./data/emission_matrix_test.json', 'r') as fp:
            emission_matrix = json.load(fp)
        print("emission matrix loaded")
    else:
        print("creating emission matrix...")
        emission_matrix = compute_emission_matrix(obs, test_corpus_df, False)
        with open('./data/emission_matrix_test.json', 'w') as fp:
            json.dump(emission_matrix, fp)
        print("emission matrix created and saved")
