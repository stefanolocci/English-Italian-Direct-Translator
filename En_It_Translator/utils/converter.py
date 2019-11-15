from conll_df import conll_df
"""
Script used to convert file with .conllu extension in .csv
"""
path = 'en_partut-ud-dev.conllu'
df = conll_df(path, file_index=False)
word_tag_df = df[['w', 'x']]
# print(word_tag_df)
word_tag_df.to_csv('../data/ud-word-pos-partut-dev.csv', sep='\t', header=False, index=False)
