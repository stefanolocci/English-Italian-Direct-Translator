from operator import itemgetter

from pos_tagging.CorpusManager import CorpusManager
from utils import config_data


class Baseline:
    def __init__(self, train_manager):
        self.train_manager = train_manager

    def compute_baseline(self, observation):
        token_obs = observation.split()
        backpointer = []
        pos_tags = config_data.get_pos_tags()
        for word in token_obs:
            freq_list = []
            if word not in self.train_manager.get_words():
                backpointer.append([word, 'NOUN'])
            else:
                for pos in pos_tags:
                    freq_list.append((pos, self.train_manager.count_word_tag_frequency(word, pos)))
                backpointer.append([word, max(freq_list, key=itemgetter(1))[0]])
        return backpointer
