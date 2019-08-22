import pandas as pd


def add_sentence_start_tag(input_file_path, output_file_path):
    train_corpus_df = pd.read_csv(input_file_path, sep='\t')

    words = list(train_corpus_df['word'])
    tags = list(train_corpus_df['tag'])
    for i, word in enumerate(words):
        if word == '.' or word == ':' or word == ';' or word == '!' or word == '?':
            words.insert(i + 1, " ")
            tags.insert(i + 1, "S0")

    word_tag_dict = {"word": words, "tag": tags}
    df = pd.DataFrame(word_tag_dict, columns=['word', 'tag'])
    df.to_csv(output_file_path, sep='\t', encoding='utf-8', header=False, index=False)


if __name__ == "__main__":
    input_path = '../data/ud-word-pos-ewt-test.csv'
    output_path = '../data/ud-word-pos-ewt-test-start.csv'
    add_sentence_start_tag(input_path, output_path)
