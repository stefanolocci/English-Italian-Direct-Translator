import json
import math
import os.path
from operator import itemgetter
from time import sleep

import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

from En_It_Translator.translator import translate_sentence
from En_It_Translator.utils import config_data
from En_It_Translator.utils.number_utility import is_ordinal_number, is_roman_number, is_number
from En_It_Translator.utils.time_it import timeit


@timeit
def compute_emission_matrix(observation, is_baseline):
    em_matrix = {}
    lemmatizer = WordNetLemmatizer()
    pos_tags = config_data.get_pos_tags()
    obs_words = observation.split()
    words = train_corpus_df['word'].str.lower().values
    for word in obs_words:
        dic = {}
        if word.lower() in words:  # word is known
            for pos in pos_tags:
                word_tag_freq = count_word_tag_frequency(word, pos)
                likelihood = word_tag_freq / count_pos_tag_frequency(pos)
                dic.update({pos: likelihood})
            em_matrix.update({word: dic})
        else:  # unknown word handling
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
    for i, pos in enumerate(state_graph):
        prob = multiply_probability(compute_transition_probability("S0", pos),
                                    (em_matrix.get(token_obs[0]).get(pos)))
        vit_matrix[i, 0] = prob
        prob_list.append([token_obs[0], pos, prob])
    backpointer.append(max(prob_list, key=itemgetter(2)))
    for j, token in enumerate(token_obs[1:]):
        prob_list = []
        w, prev_max_pos, prev_max_prob = backpointer[len(backpointer) - 1]
        for k, pos in enumerate(state_graph):
            prob = multiply_probability(float(prev_max_prob),
                                        multiply_probability(compute_transition_probability(
                                            prev_max_pos, pos),
                                            em_matrix.get(token).get(pos)))
            vit_matrix[k, j + 1] = prob
            prob_list.append([token, pos, prob])
        backpointer.append(max(prob_list, key=itemgetter(2)))
    return backpointer


def run_translator():
    s1 = "The black droid then lowers Vader 's mask and helmet onto his head "
    s2 = "These are not the droids you 're looking for "
    s3 = "Your friends may escape , but you are doomed"
    obs = s1 + s2 + s3
    emission_matrix = get_emission_matrix('./data/emission_matrix_train.json', obs)
    vit = refine_result(viterbi(s1, emission_matrix))
    vit1 = refine_result(viterbi(s2, emission_matrix))
    vit2 = refine_result(viterbi(s3, emission_matrix))
    for w, tag in zip(s1.split(), vit):
        print("{} <--- {}".format(w, tag[1]))
    print("\n****************\n")
    for w, tag in zip(s2.split(), vit1):
        print("{} <--- {}".format(w, tag[1]))
    print("\n****************\n")
    for w, tag in zip(s3.split(), vit2):
        print("{} <--- {}".format(w, tag[1]))
    print(translate_sentence(vit))
    print(translate_sentence(vit1))
    print(translate_sentence(vit2))


def compute_accuracy(predicted_tags, real_tags):
    counter = 0
    for pt, rt, in zip(predicted_tags, real_tags):
        if not pt[1] == rt:
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
        emission_matrix = compute_emission_matrix(observation, False)
        with open(path, 'w') as fp:
            json.dump(emission_matrix, fp)
        print("emission matrix created and saved")
    return emission_matrix


def compute_baseline(observation, baseline_emission_matrix):
    token_obs = observation.split()
    backpointer = []
    for word in token_obs:
        backpointer.append([word, max(baseline_emission_matrix.get(word).items(), key=itemgetter(1))[0]])
    return backpointer


def refine_result(pos_tag_result):
    punct_char = [",", ".", ":", ";", "[", "]", "{", "}", "(", ")", "?", "!"]
    prev_word, prev_tag = '', ''
    for i, res in enumerate(pos_tag_result):
        curr_word, curr_tag = res[0], res[1]
        if i > 0:
            prev_word, prev_tag = pos_tag_result[i - 1][0], pos_tag_result[i - 1][1]
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
        elif prev_word != '' and curr_word[0].isupper() and len(curr_word) > 2 and prev_tag != 'PUNCT' \
                and prev_word[0].islower():
            res[1] = 'PROPN'
    return pos_tag_result


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def test_viterbi():
    test_emission_matrix = get_emission_matrix('./data/emission_matrix_test.json', observ)
    viterbi_result = []
    progr_bar_length = len(sentences)

    print_progress_bar(0, progr_bar_length, prefix='Progress:', suffix='Complete', length=50)
    for index, sentence in enumerate(sentences):
        sleep(0.1)
        print_progress_bar(index + 1, progr_bar_length, prefix='Progress:', suffix='Complete', length=50)
        viterbi_result = viterbi_result + viterbi(sentence, test_emission_matrix)
    viterbi_result = refine_result(viterbi_result)
    print("Viterbi accuracy: {}".format(compute_accuracy(viterbi_result, test_corpus_df_check['tag'].tolist())))


def test_baseline():
    baseline_emission_matrix = get_emission_matrix('./data/emission_matrix_test_baseline.json', observ)
    baselilne_result = []
    for sentence in sentences:
        baselilne_result = baselilne_result + compute_baseline(sentence, baseline_emission_matrix)
    print("Baseline accuracy: {}".format(compute_accuracy(baselilne_result, test_corpus_df_check['tag'].tolist())))


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

    run_translator()
    # test_viterbi()
    # test_baseline()
