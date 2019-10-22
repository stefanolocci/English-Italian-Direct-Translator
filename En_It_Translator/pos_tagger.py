import io
import json
import math
import os.path
import re
from operator import itemgetter

import numpy as np
import pandas as pd

from translator import translate_sentence
from utils import config_data
from utils.config_data import get_pos_tags
from utils.number_utility import is_ordinal_number, is_number, is_roman_number
from utils.progress_bar import print_progress_bar
from utils.time_it import timeit


def check_word_suffix(word):
    if any(word.lower().endswith(suffix) for suffix in config_data.noun_suffix):
        return 'NOUN'
    elif any(word.lower().endswith(suffix) for suffix in config_data.adj_suffix):
        return 'ADJ'
    elif any(word.lower().endswith(suffix) for suffix in config_data.adv_suffix):
        return 'ADV'
    elif any(word.lower().endswith(suffix) for suffix in config_data.verb_suffix):
        return 'VERB'
    else:
        return None


@timeit
def compute_emission_matrix(observation):
    em_matrix = {}
    pos_tags = config_data.get_pos_tags()
    obs_words = observation.split()
    pos_smooth = 1 / len(get_pos_tags())
    obs_length = len(obs_words)
    for i, word in enumerate(obs_words):
        if i % 500 == 0:
            print("{} processed words of {} total words".format(i, obs_length))
        dic = {}
        if word in train_words:  # word is known
            for pos in pos_tags:
                word_tag_freq = count_word_tag_frequency(word, pos)
                likelihood = word_tag_freq / count_pos_tag_frequency(pos)
                dic.update({pos: likelihood})
            em_matrix.update({word: dic})
        elif word in dev_words:
            for pos in pos_tags:
                word_tag_freq = count_word_tag_frequency_dev(word, pos)
                likelihood = word_tag_freq / count_pos_tag_frequency_dev(pos)
                dic.update({pos: likelihood})
            em_matrix.update({word: dic})
        elif check_word_suffix(word):
            suffix_predicted_pos = check_word_suffix(word)
            for pos in pos_tags:
                if pos == suffix_predicted_pos:
                    dic.update({pos: 0.5})
                else:
                    dic.update({pos: 0.0})
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
def count_word_tag_frequency(wrd, pos):
    return interleaved_w_t.count(" " + wrd.lower() + " " + pos)


# conta generale dei tag
def count_pos_tag_frequency_dev(pos_tag):
    return corpus_tag_frequencies_dev[pos_tag]


# conta occorenze parola taggata con un certo pos
def count_word_tag_frequency_dev(wrd, pos):
    return interleaved_w_t_dev.count(" " + wrd.lower() + " " + pos)


# conta occorrenze di due tag che si susseguono
def count_tags_co_occurrence(previous_tag, current_tag):
    return train_tags_start.count(previous_tag + " " + current_tag)


# calcola la probabilità di transizione, ovvero la probabilità che dato un tag (previous tag), il successivo
# sia il tag current tag
def compute_transition_probability(previous_tag, current_tag):
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
    max_prob = 0
    max_pos = ""
    for i, pos in enumerate(state_graph):
        prob = multiply_probability(compute_transition_probability("S0", pos),
                                    (em_matrix.get(token_obs[0]).get(pos))) + config_data.laplace_smoothing
        vit_matrix[i, 0] = prob
        if prob > max_prob:
            max_pos = pos
            max_prob = prob
    backpointer.append([token_obs[0], max_pos, vit_matrix[:, 0].max()])
    for j, token in enumerate(token_obs[1:]):
        w_, prev_max_pos, prev_max_prob = backpointer[-1]
        max_prob = 0
        max_pos = ""
        for k, pos in enumerate(state_graph):
            temp_prob = multiply_probability(float(prev_max_prob), compute_transition_probability(prev_max_pos, pos))
            prob = multiply_probability(em_matrix.get(token).get(pos), temp_prob) + config_data.laplace_smoothing
            vit_matrix[k, j + 1] = prob
            if prob > max_prob:
                max_pos = pos
                max_prob = prob
        backpointer.append([token, max_pos, vit_matrix[:, j + 1].max()])
    return backpointer


def run_translator():
    observations = ""
    with open('./data/sentences.txt') as test_sentences:
        for line in test_sentences:
            observations += line.strip()
        emission_matrix = get_emission_matrix('./data/sentence-emission-matrix-noun-suffix-smooth.json',
                                              " ".join(observations.split(sep=';')))
        for i, sent in enumerate(observations.split(sep=';')):
            vit_res = refine_result(viterbi(sent, emission_matrix))
            for w, tag in zip(sent.split(), vit_res):
                print("{} <--- {}".format(w, tag[1]))
            print(translate_sentence(vit_res, ("d" + str(i + 1))))
            print("\n****************\n")


def compute_accuracy(predicted_tags, real_tags):
    counter = 0
    for i, pt in enumerate(predicted_tags):
        if pt[1] != real_tags[i]:
            print("{}, {}, {}".format(pt[0], pt[1], real_tags[i]))
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
        if word not in train_words:
            backpointer.append([word, 'NOUN'])
        else:
            # backpointer.append([word, word_freq_dict.get(word).get(0)])
            for pos in pos_tags:
                freq_list.append((pos, count_word_tag_frequency(word, pos)))

            backpointer.append([word, max(freq_list, key=itemgetter(1))[0]])
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
        # elif prev_word != '' and curr_word[0].isupper() and len(
        #         curr_word) > 2 and curr_word[1].islower() and prev_tag != 'PUNCT' and curr_tag != 'ADJ':
        #     res[1] = 'PROPN'
        # elif curr_word[0].isupper() and len(curr_word) > 2 and curr_word[1].islower() and prev_tag and (
        #         prev_word == '' or prev_tag == 'PUNCT') and curr_tag == 'NOUN':
        #     res[1] = 'PROPN'
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
    test_emission_matrix = get_emission_matrix('./data/' + config_data.dataset +
                                               '/emission_matrix/emission_matrix_test_' +
                                               config_data.dataset + '-noun-suffix-smooth.json', observ)
    progr_bar_length = len(sentences)
    print_progress_bar(0, progr_bar_length, prefix='Progress:', suffix='Complete', length=50)
    viterbi_result = []
    for pb_index, sent in enumerate(sentences):
        print_progress_bar(pb_index + 1, progr_bar_length, prefix='Progress:', suffix='Complete', length=50)
        viterbi_result += refine_result(viterbi(sent, test_emission_matrix))
    print("Viterbi accuracy mean: {}".format(compute_accuracy(viterbi_result, test_tags_list)))


@timeit
def test_baseline():
    progr_bar_length = len(sentences)
    print_progress_bar(0, progr_bar_length, prefix='Progress:', suffix='Complete', length=50)
    baseline_result = []
    for pb_index, sent in enumerate(sentences):
        print_progress_bar(pb_index + 1, progr_bar_length, prefix='Progress:', suffix='Complete', length=50)
        baseline_result += compute_baseline(sent)
    print("Baseline accuracy mean: {}".format(compute_accuracy(baseline_result, test_tags_list)))


if __name__ == "__main__":
    # train data initialization
    train_words = []
    train_tag_list = []
    interleaved_w_t = ""
    with io.open(config_data.training_set_path, encoding='utf-8') as train_file:
        next(train_file)
        for line in train_file:
            data = line.strip().split(sep='\t')
            train_words.append(data[0])
            train_tag_list.append(data[1])
            interleaved_w_t += data[0].lower() + " " + data[1] + " "
    train_tag_list_start = [val for val in train_tag_list]
    for index, (word, tag) in enumerate(zip(train_words, train_tag_list_start)):
        if word == '.' or word == ';' or word == '!' or word == '?':
            train_tag_list_start.insert(index + 1, "S0")
    train_tag_list_start.insert(0, "S0")
    train_tags_start = " ".join([str(t) for t in train_tag_list_start])
    train_tag_list_start_df = pd.DataFrame(train_tag_list_start, columns=['tag'])
    corpus_tag_frequencies = train_tag_list_start_df['tag'].value_counts()

    # test data initialization
    test_words = []
    test_tags_list = []
    with io.open(config_data.test_set_path, encoding='utf-8') as test_file:
        next(test_file)
        for line in test_file:
            data = line.strip().split(sep='\t')
            test_words.append(data[0])
            test_tags_list.append(data[1])

    test_tag_list_start = [val for val in test_tags_list]
    for index, (word, tag) in enumerate(zip(test_words, test_tag_list_start)):
        if word == '.' or word == ';' or word == '!' or word == '?':
            test_tag_list_start.insert(index + 1, "S0")
    test_tag_list_start.insert(0, "S0")
    # print(test_tag_list_start)
    observ = " ".join(test_words)
    sentences = []
    sentence = ""
    for index, w in enumerate(test_words):
        if test_tag_list_start[index + 1] != 'S0':
            sentence += w + " "
        else:
            if w in [')', '.', '?', '-', '--', '----', ']', '...', '..']:
                sentence += w + " "
                sentences.append(sentence)
                sentence = ""
            else:
                sentences.append(sentence)
                sentence = w + " "
    sentences = list(filter(''.__ne__, sentences))

    dev_words = []
    dev_tag_list = []
    interleaved_w_t_dev = ""
    with io.open(config_data.dev_set_path, encoding='utf-8') as dev_file:
        next(dev_file)
        for line in dev_file:
            data = line.strip().split(sep='\t')
            dev_words.append(data[0])
            dev_tag_list.append(data[1])
            interleaved_w_t_dev += data[0].lower() + " " + data[1] + " "

    dev_tag_list_start = [val for val in dev_tag_list]
    for index, (word, tag) in enumerate(zip(dev_words, dev_tag_list_start)):
        if word == '.' or word == ';' or word == '!' or word == '?':
            dev_tag_list_start.insert(index + 1, "S0")
    dev_tag_list_start.insert(0, "S0")
    dev_tags_start = " ".join([str(t) for t in dev_tag_list_start])
    dev_tag_list_start_df = pd.DataFrame(train_tag_list_start, columns=['tag'])
    corpus_tag_frequencies_dev = dev_tag_list_start_df['tag'].value_counts()

    run_translator()
    # test_viterbi()
    # test_baseline()
