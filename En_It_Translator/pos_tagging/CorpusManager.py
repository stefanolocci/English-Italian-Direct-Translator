import io

import pandas as pd

from utils import config_data
from utils.time_it import timeit


class CorpusManager:

    def __init__(self):
        self.words = []
        self.tag_list = []
        self.tag_list_start = ""
        self.corpus_tag_frequencies = None
        self.sentences = []
        self.word_freqs = {}

    @timeit
    def read_corpus(self, path):
        with io.open(path, encoding='utf-8') as file:
            next(file)
            sentence = ""
            for line in file:
                data = line.strip().split(sep='\t')
                self.words.append(data[0])
                sentence += data[0] + " "
                self.tag_list.append(data[1])
                if data[0] not in config_data.sentence_split_sep:
                    self.tag_list_start += data[1] + " "
                else:
                    self.tag_list_start += data[1] + " "
                    self.tag_list_start += "S0 "
                    self.sentences.append(sentence)
                    sentence = ""
                word_tag = (data[0].lower(), data[1])
                res = self.word_freqs.get(word_tag)
                if res:
                    self.word_freqs[word_tag] += 1
                else:
                    self.word_freqs[word_tag] = 1
        self.corpus_tag_frequencies = self.__get_tag_frequencies()
        return self

    def __get_tag_frequencies(self):
        train_tag_list_start_df = pd.DataFrame(self.tag_list_start.split(), columns=['tag'])
        return train_tag_list_start_df['tag'].value_counts()

    # conta occorenze parola taggata con un certo pos
    def count_word_tag_frequency(self, wrd, pos):
        res = self.word_freqs.get((wrd.lower(), pos))
        if res:
            return res
        else:
            return 0

    def count_pos_tag_frequency(self, pos_tag):
        return self.corpus_tag_frequencies[pos_tag]

    def count_tags_co_occurrence(self, previous_tag, current_tag):
        return self.tag_list_start.count(previous_tag + " " + current_tag)

    def preprocess_observations(self):
        return self.sentences

    def get_words(self):
        return self.words

    def get_tags(self):
        return self.tag_list

    def check_word_suffix(self, word):
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
