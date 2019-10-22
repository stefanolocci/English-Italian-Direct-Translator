import io

import pandas as pd
import numpy as np


def add_sentence_start_tag(input_file_path, output_file_path):
    train_corpus_df = pd.read_csv(input_file_path, sep='\t')
    train_corpus_df = train_corpus_df.replace(np.nan, "", regex=True)

    words = list(train_corpus_df['word'])
    tags = list(train_corpus_df['tag'])
    for i, word in enumerate(words):
        if word == '.' or word == ';' or word == '!' or word == '?':
            words.insert(i + 1, " ")
            tags.insert(i + 1, "S0")

    word_tag_dict = {"word": words, "tag": tags}
    df = pd.DataFrame(word_tag_dict, columns=['word', 'tag'])
    df.to_csv(output_file_path, sep='\t', header=False, index=False)


def add_start_txt(ip, op):
    words = []
    tags = []
    with io.open(ip, encoding='utf-8') as f:
        for line in f:
            split = line.strip().split(sep='\t')
            words.append(split[0])
            tags.append(split[1])
    for index, (word, tag) in enumerate(zip(words, tags)):
        if word == '.' or word == ';' or word == '!' or word == '?':
            words.insert(index + 1, "")
            tags.insert(index + 1, "S0")
    word_tag_dict = {"word": words, "tag": tags}
    df = pd.DataFrame(word_tag_dict, columns=['word', 'tag'])
    df.to_csv(op, sep='\t', header=False, index=False)
    return df


if __name__ == "__main__":
    # input_path = '../data/ud-word-pos-gum-train.csv'
    # output_path = '../data/ud-word-pos-gum-train-start.csv'
    input_path = '../data/train.txt'
    output_path = '../data/train_start.txt'
    # add_sentence_start_tag(input_path, output_path)
    add_start_txt(input_path, output_path)
