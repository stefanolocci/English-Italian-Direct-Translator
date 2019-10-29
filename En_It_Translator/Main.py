import re

from Translator import Translator
from pos_tagging.Baseline import Baseline
from pos_tagging.EmissionMatrix import EmissionMatrix
from pos_tagging.CorpusManager import CorpusManager
from pos_tagging.EvalPosTagger import EvalPosTagger
from pos_tagging.Viterbi import Viterbi
from utils import config_data, number_utility
from utils.progress_bar import print_progress_bar
from utils.time_it import timeit


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
        elif number_utility.is_number(curr_word) and curr_tag != 'NUM':
            res[1] = 'NUM'
        elif 'th' in curr_word and number_utility.is_ordinal_number(curr_word):
            res[1] = 'NOUN'
        elif number_utility.is_roman_number(curr_word) and curr_tag != 'PRON' and curr_tag != 'PROPN':
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
def test_viterbi(observations):
    emission_matrix_path = './data/' + config_data.dataset + '/emission_matrix/emission_matrix_test_' + \
                           config_data.dataset + '-noun-smooth.json'
    sentences = test_manager.preprocess_observations()
    test_emission_matrix = em_matrix.get_emission_matrix(emission_matrix_path, observations)
    progr_bar_length = len(sentences)
    print_progress_bar(0, progr_bar_length, prefix='Progress:', suffix='Complete', length=50)
    viterbi_result = []
    for pb_index, sent in enumerate(sentences):
        print_progress_bar(pb_index + 1, progr_bar_length, prefix='Progress:', suffix='Complete', length=50)
        viterbi_result += refine_result(viterbi.viterbi(sent, test_emission_matrix))
    print("Viterbi accuracy mean: {}".format(evaluator.compute_accuracy(viterbi_result)))


@timeit
def test_baseline():
    sentences = test_manager.preprocess_observations()
    progr_bar_length = len(sentences)
    print_progress_bar(0, progr_bar_length, prefix='Progress:', suffix='Complete', length=50)
    baseline_result = []
    for pb_index, sent in enumerate(sentences):
        print_progress_bar(pb_index + 1, progr_bar_length, prefix='Progress:', suffix='Complete', length=50)
        baseline_result += baseline.compute_baseline(sent)
    print("Baseline accuracy mean: {}".format(evaluator.compute_accuracy(baseline_result)))


def run_translator():
    observations = ""
    with open('./data/sentences.txt') as test_sentences:
        for line in test_sentences:
            observations += line.strip()
        emission_matrix = em_matrix.get_emission_matrix('./data/sentence-emission-matrix-noun-suffix-smooth.json',
                                                        " ".join(observations.split(sep=';')))
        for i, sent in enumerate(observations.split(sep=';')):
            vit_res = refine_result(viterbi.viterbi(sent, emission_matrix))
            for w, tag in zip(sent.split(), vit_res):
                print("{} <--- {}".format(w, tag[1]))
            print(translator.translate_sentence(vit_res, ("d" + str(i + 1))))
            print("\n****************\n")


if __name__ == '__main__':
    print("Reading training, test and development corpus...")
    train_manager = CorpusManager().read_corpus(config_data.training_set_path)
    test_manager = CorpusManager().read_corpus(config_data.test_set_path)
    dev_manager = CorpusManager().read_corpus(config_data.dev_set_path)
    print("Done!")

    translator = Translator()
    viterbi = Viterbi(train_manager)
    baseline = Baseline(train_manager)
    em_matrix = EmissionMatrix(train_manager, dev_manager)
    evaluator = EvalPosTagger(test_manager)

    test_observations = " ".join(test_manager.words)
    # run_translator()
    test_viterbi(test_observations)
    test_baseline()
