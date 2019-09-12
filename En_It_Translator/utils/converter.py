import pandas as pd
from conll_df import conll_df

path = '../data/en_lines-ud-train.conllu'
df = conll_df(path, file_index=False)
# word_tag_df = df[['w', 'x']]
# print(word_tag_df)
# word_tag_df.to_csv('ud-word-pos-ewt-train.csv', sep='\t',header=False,index=False)
