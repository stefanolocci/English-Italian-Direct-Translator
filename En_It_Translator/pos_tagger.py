import json
import math
import os.path
import re
from operator import itemgetter

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize

from translator import translate_sentence
from utils import config_data
from utils.config_data import get_pos_tags
from utils.number_utility import is_ordinal_number, is_number, is_roman_number
from utils.progress_bar import print_progress_bar
from utils.time_it import timeit


@timeit
def compute_emission_matrix(observation):
    em_matrix = {}
    pos_tags = config_data.get_pos_tags()
    obs_words = observation.split()
    words = train_corpus_df['word'].str.lower().values
    pos_smooth = 1 / len(get_pos_tags())
    for word in obs_words:
        dic = {}
        if word.lower() in words:  # word is known
            for pos in pos_tags:
                word_tag_freq = count_word_tag_frequency(word, pos)
                likelihood = word_tag_freq / count_pos_tag_frequency(pos)
                dic.update({pos: likelihood})
            em_matrix.update({word: dic})
        else:  # unknown word handling
            # for pos in pos_tags:
            #     dic.update({pos: pos_smooth})
            for pos in pos_tags:
                if pos == 'NOUN':
                    dic.update({pos: 1.0})
                else:
                    dic.update({pos: 0.0})
        em_matrix.update({word: dic})
    return em_matrix


# conta generale dei tag
def count_pos_tag_frequency(pos_tag):
    return corpus_tag_frequencies[pos_tag]


# conta occorenze parola taggata con un certo pos
def count_word_tag_frequency(word, pos):
    word_tag = " " + word.lower() + " " + pos
    return interleaved_w_t.count(word_tag)


# conta occorrenze di due tag che si susseguono
def count_tags_co_occurrence(previous_tag, current_tag):
    tag_tag = previous_tag + " " + current_tag
    return train_tags.count(tag_tag)


# calcola la probabilità di transizione, ovvero la probabilità che dato un tag (previous tag), il successivo
# sia il tag current tag
def compute_transition_probability(previous_tag, current_tag):
    # return count_tags_co_occurrence(previous_tag, current_tag) / count_pos_tag_frequency(previous_tag)
    return count_tags_co_occurrence(previous_tag, current_tag) / count_pos_tag_frequency(previous_tag)


def multiply_probability(p1, p2):
    if p1 == 0 or p2 == 0:
        return 0.0
    return math.exp(math.log(p1) + math.log(p2))


# state graph: lista di pos tags
# @timeit
def viterbi(observation, em_matrix):
    state_graph = config_data.get_pos_tags()
    backpointer = []
    token_obs = observation.split()
    vit_matrix = np.zeros((len(state_graph), len(token_obs)))
    prob_list = []
    for i, pos in enumerate(state_graph):  # initial step S0 ->
        prob = multiply_probability(compute_transition_probability("S0", pos),
                                    (em_matrix.get(token_obs[0]).get(pos)))
        vit_matrix[i, 0] = prob
        prob_list.append([token_obs[0], pos, prob])
    backpointer.append(max(prob_list, key=itemgetter(2)))
    for j, token in enumerate(token_obs[1:]):
        prob_list = []
        w, prev_max_pos, prev_max_prob = backpointer[-1]
        for k, pos in enumerate(state_graph):
            prob = multiply_probability(float(prev_max_prob),
                                        multiply_probability(compute_transition_probability(prev_max_pos, pos),
                                                             em_matrix.get(token).get(pos)))
            vit_matrix[k, j + 1] = prob
            prob_list.append([token, pos, prob])
        backpointer.append(max(prob_list, key=itemgetter(2)))
    return backpointer


def run_translator():
    observations = ""
    with open('./data/sentences.txt') as test_sentences:
        for line in test_sentences:
            observations += line.strip()
        emission_matrix = get_emission_matrix('./data/emission_matrix_train.json',
                                              " ".join(observations.split(sep=';')))
        for i, sent in enumerate(observations.split(sep=';')):
            vit_res = refine_result(viterbi(sent, emission_matrix))
            for w, tag in zip(sent.split(), vit_res):
                print("{} <--- {}".format(w, tag[1]))
            print(translate_sentence(vit_res, ("d" + str(i + 1))))
            print("\n****************\n")


def compute_accuracy(predicted_tags, real_tags):
    counter = 0
    for index, (pt, rt) in enumerate(zip(predicted_tags, real_tags)):
        if not pt[1] == rt:
            print("{}, {}, {}".format(pt[0], pt[1], rt))
            counter += 1
    return (len(real_tags) - counter) / len(real_tags)


def get_emission_matrix(path, observation):
    if os.path.exists(path):
        print("loading emission matrix...")
        with open(path, 'r') as fp:
            emission_matrix = json.load(fp)
        print("emission matrix loaded")
    else:
        print("creating emission matrix...")
        emission_matrix = compute_emission_matrix(observation)
        with open(path, 'w') as fp:
            json.dump(emission_matrix, fp)
        print("emission matrix created and saved")
    return emission_matrix


def compute_baseline(observation):
    token_obs = observation.split()
    backpointer = []
    pos_tags = get_pos_tags()
    for word in token_obs:
        freq_list = []
        if word.lower() not in word_list:
            backpointer.append([word, 'NOUN'])
        else:
            backpointer.append([word, word_freq_dict.get(word).get(0)])
            # for pos in pos_tags:
            #     freq_list.append((pos, count_word_tag_frequency(word, pos)))
            # backpointer.append([word, max(freq_list, key=itemgetter(1))[0]])
    return backpointer


def refine_result(pos_tag_result):
    punct_char = [",", ".", ":", ";", "[", "]", "{", "}", "(", ")", "?", "!"]
    prev_word, prev_tag = '', ''
    next_word, next_tag = '', ''
    next_next_word, next_next_tag = '', ''
    next_next_next_word, next_next_next_tag = '', ''
    pattern_date = '[0-9]{2}/[0-9]{2}/[0-9]{4}'
    pattern_time = '([0-9]+:[0-9]+)'
    pattern_tel = '([0-9]+-[0-9]+)'
    pattern_num_comma = '([0-9]+,[0-9]+)'
    pattern_num_dot = '([0-9]+.[0-9]+)'
    pattern_coord = '([EY][0-9]+\.[0-9]+)'
    for i, res in enumerate(pos_tag_result):
        curr_word, curr_tag = res[0], res[1]
        if i > 0:
            prev_word, prev_tag = pos_tag_result[i - 1][0], pos_tag_result[i - 1][1]
        if i < len(pos_tag_result) - 1:
            next_word, next_tag = pos_tag_result[i + 1][0], pos_tag_result[i + 1][1]
        if i < len(pos_tag_result) - 2:
            next_next_word, next_next_tag = pos_tag_result[i + 2][0], pos_tag_result[i + 2][1]
        if i < len(pos_tag_result) - 3:
            next_next_next_word, next_next_next_tag = pos_tag_result[i + 3][0], pos_tag_result[i + 3][1]
        if curr_word in punct_char and curr_tag != 'PUNCT':
            res[1] = 'PUNCT'
        elif 'http' in curr_word or '.com' in curr_word or '@' in curr_word:
            res[1] = 'X'
        elif is_number(curr_word) and curr_tag != 'NUM':
            res[1] = 'NUM'
        elif 'th' in curr_word and is_ordinal_number(curr_word):
            res[1] = 'NOUN'
        elif is_roman_number(curr_word) and curr_tag != 'PRON' and curr_tag != 'PROPN':
            res[1] = 'NUM'
        elif prev_word != '' and curr_word[0].isupper() and len(
                curr_word) > 2 and curr_word[1].islower() and prev_tag != 'PUNCT' and curr_tag != 'ADJ':
            res[1] = 'PROPN'
        elif curr_word[0].isupper() and len(curr_word) > 2 and curr_word[1].islower() and prev_tag and (
                prev_word == '' or prev_tag == 'PUNCT') and curr_tag == 'NOUN':
            res[1] = 'PROPN'
        elif curr_word == 'to' and (
                next_tag == 'PROPN' or next_tag == 'NOUN' or next_tag == 'PRON' or next_tag == 'DET'):
            res[1] = 'ADP'
        elif curr_word == 'to' and (next_tag == 'AUX' or next_tag == 'VERB'):
            res[1] = 'PART'
        elif (curr_word.lower() == 'have' or curr_word.lower() == 'are' or curr_word.lower() == 'had'
              or curr_word.lower() == 'was' or curr_word.lower() == 'has') \
                and (next_tag == 'AUX' or next_tag == 'VERB'):
            res[1] = 'AUX'
        elif curr_word.lower() == 'that' and (
                next_tag == 'NOUN' or next_next_tag == 'NOUN' or next_next_next_tag == 'NOUN'):
            res[1] = 'SCONJ'
        elif curr_word.lower() == 'that' and (
                next_tag != 'NOUN' and next_next_tag != 'NOUN' and next_next_next_tag != 'NOUN'):
            res[1] = 'PRON'
        elif re.match(pattern_date, curr_word) or re.match(pattern_time, curr_word) \
                or re.match(pattern_tel, curr_word) or re.match(pattern_num_comma, curr_word) \
                or re.match(pattern_coord, curr_word) or re.match(pattern_num_dot, curr_word):
            res[1] = 'NUM'
        elif curr_word == 'not' and (next_tag == 'PROPN' or next_tag == 'NOUN' or next_tag == 'PRON'):
            res[1] = 'ADV'
        elif curr_word == 'not' and (next_tag == 'AUX' or next_tag == 'VERB'):
            res[1] = 'PART'
        elif curr_word == 'of' and next_word == 'course':
            res[1] = 'ADV'
        elif curr_word == 'course' and prev_word == 'of':
            res[1] = 'ADV'
        elif curr_word == 'most' and next_tag == 'ADJ':
            res[1] = 'ADV'
        elif curr_word == 'most' and next_tag == 'ADP':
            res[1] = 'ADJ'
        elif '-' in curr_word and '=' in curr_word:
            res[1] = 'SYM'
        elif '------' in curr_word:
            res[1] = 'PUNCT'
    return pos_tag_result


@timeit
def test_viterbi():
    test_emission_matrix = get_emission_matrix('./data/emission_matrix_test.json', observ)
    viterbi_result = []
    progr_bar_length = len(sentences)
    print_progress_bar(0, progr_bar_length, prefix='Progress:', suffix='Complete', length=50)
    for index, sentence in enumerate(sentences):
        print_progress_bar(index + 1, progr_bar_length, prefix='Progress:', suffix='Complete', length=50)
        viterbi_result = viterbi_result + viterbi(sentence, test_emission_matrix)
    viterbi_result = refine_result(viterbi_result)
    print("Viterbi accuracy: {}".format(compute_accuracy(viterbi_result, test_corpus_df_check['tag'].tolist())))


@timeit
def test_baseline():
    baselilne_result = []
    progr_bar_length = len(sentences)
    print_progress_bar(0, progr_bar_length, prefix='Progress:', suffix='Complete', length=50)
    for index, sentence in enumerate(sentences):
        print_progress_bar(index + 1, progr_bar_length, prefix='Progress:', suffix='Complete', length=50)
        baselilne_result = baselilne_result + compute_baseline(sentence)
    print("Baseline accuracy: {}".format(compute_accuracy(baselilne_result, test_corpus_df_check['tag'].tolist())))


# def compute_word_pos_frequency_table():
#     word_freq_dict = {}
#     wuords = train_corpus_df['word'].str.lower().tolist()
#     for word in words_to_test:
#         if word.lower() in wuords:
#             s = train_corpus_df.loc[train_corpus_df['word'].str.lower() ==
#               word.lower()]['tag'].value_counts().nlargest(
#                 1).index[0]
#             word_freq_dict.update({word: [s]})
#     df = pd.DataFrame(word_freq_dict).to_csv('./data/word_frequencies.csv')


if __name__ == "__main__":
    # train data initialization
    training_set_path = config_data.training_set_path
    train_corpus_df = pd.read_csv(training_set_path, sep='\t')
    train_tag_list = train_corpus_df['tag'].tolist()
    train_tags = " ".join(train_tag_list)
    corpus_tag_frequencies = train_corpus_df['tag'].value_counts()
    word_list = train_corpus_df['word'].str.lower().tolist()
    interleaved_w_t = " ".join([val for pair in zip(word_list, train_tag_list) for val in pair])

    # test data initialization
    test_set_path = config_data.test_set_path
    test_corpus_df = pd.read_csv(test_set_path, sep='\t')
    words_to_test = list(filter(" ".__ne__, test_corpus_df['word']))
    observ = " ".join(words_to_test)
    sentences = sent_tokenize(observ)
    test_set_path_check = config_data.test_set_path_check
    test_corpus_df_check = pd.read_csv(test_set_path_check, sep='\t')

    word_freq_dict = pd.read_csv('./data/word_frequencies.csv').to_dict()

    # run_translator()
    test_viterbi()
    # test_baseline()
