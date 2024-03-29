import json
import os

from utils import config_data
from utils.time_it import timeit


class EmissionMatrix:

    def __init__(self, train_manager, dev_manager):
        self.train_manager = train_manager
        self.dev_manager = dev_manager

    def get_emission_matrix(self, path, observation):
        """
        Check if the emission matrix exists, if not creates and save it
        :param path: system path where the emission matrix is saved
        :param observation: list of word
        :return: emission matrix
        """
        if os.path.exists(path):
            print("loading emission matrix...")
            with open(path, 'r') as fp:
                emission_matrix = json.load(fp)
            print("emission matrix loaded")
        else:
            print("creating emission matrix...")
            emission_matrix = self.__compute_emission_matrix(observation)
            with open(path, 'w') as fp:
                json.dump(emission_matrix, fp)
            print("emission matrix created and saved")
        return emission_matrix

    @timeit
    def __compute_emission_matrix(self, observation):
        """
        Method to compute the emission matrix
        :param observation: list of word of which compute the emission probability
        :return: matrix containing the emission probability for each pos tag for each word of the observation
        """
        em_matrix = {}
        pos_tags = config_data.get_pos_tags()
        obs_words = observation.split()
        pos_smooth = 1 / len(config_data.get_pos_tags())
        obs_length = len(obs_words)
        for i, word in enumerate(obs_words):
            if i % 500 == 0:
                print("{} processed words of {} total words".format(i, obs_length))
            dic = {}
            suffix_tag = self.train_manager.check_word_suffix(word)
            if word in self.train_manager.get_words():  # word is known
                for pos in pos_tags:
                    word_tag_freq = self.train_manager.count_word_tag_frequency(word, pos)
                    likelihood = word_tag_freq / self.train_manager.count_pos_tag_frequency(pos)
                    dic.update({pos: likelihood})
                em_matrix.update({word: dic})
            elif word in self.dev_manager.get_words():
                for pos in pos_tags:
                    if pos in self.dev_manager.get_tags():
                        word_tag_freq = self.dev_manager.count_word_tag_frequency(word, pos)
                        likelihood = word_tag_freq / self.dev_manager.count_pos_tag_frequency(pos)
                        dic.update({pos: likelihood})
                em_matrix.update({word: dic})
            elif suffix_tag is not None:
                for pos in pos_tags:
                    if pos == suffix_tag:
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
