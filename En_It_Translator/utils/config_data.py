dataset = 'wsj'
training_set_path = './data/' + dataset + '/ud-word-pos-' + dataset + '-train.csv'
test_set_path = './data/' + dataset + '/ud-word-pos-' + dataset + '-test.csv'
dev_set_path = './data/' + dataset + '/ud-word-pos-' + dataset + '-dev.csv'

laplace_smoothing = 0.0000000000000001

NOUN_SMOOTH = '-noun-smooth'
PROPN_SMOOTH = '-propn-smooth'
EQUAL_SMOOTH = '-equal-smooth'
DEV_SUFFIX_SMOOTH = '-noun-suffix-smooth'

noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling",
               "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize", "ing", "ed"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]

sentence_split_sep = ['!', '.', '?']


def get_pos_tags():
    return ["NOUN", "PUNCT", "VERB", "PRON", "ADP", "DET", "PROPN", "ADJ", "AUX", "ADV", "CCONJ", "PART", "NUM",
            "SCONJ", "X", "INTJ", "SYM"]
