training_set_path = './data/ud-word-pos-ewt-train-start.csv'
test_set_path = './data/ud-word-pos-ewt-test-start.csv'
test_set_path_check = './data/ud-word-pos-ewt-test.csv'
dev_set_path = './data/ud-word-pos-ewt-dev.csv'


def get_pos_tags():
    return ["NOUN", "PUNCT", "VERB", "PRON", "ADP", "DET", "PROPN", "ADJ", "AUX", "ADV", "CCONJ", "PART", "NUM",
            "SCONJ", "X", "INTJ", "SYM"]
