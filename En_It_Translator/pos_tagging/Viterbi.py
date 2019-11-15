import math

import numpy as np

from utils import config_data


class Viterbi:
    def __init__(self, train_manager):
        self.train_manager = train_manager

    def viterbi(self, observation, em_matrix):
        """
        Method that implements the Viterbi Algorithm
        :param observation: Observed sentence to pos tag
        :param em_matrix: pre-saved emission matrix
        :return: pos tagged observation
        """
        state_graph = config_data.get_pos_tags()
        backpointer = []
        token_obs = observation.split()
        vit_matrix = np.zeros((len(state_graph), len(token_obs)))
        max_prob = 0
        max_pos = ""
        for i, pos in enumerate(state_graph):
            prob = self.__multiply_probability(self.__compute_transition_probability("S0", pos),
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
                temp_prob = self.__multiply_probability(float(prev_max_prob),
                                                        self.__compute_transition_probability(prev_max_pos, pos))
                prob = self.__multiply_probability(em_matrix.get(token).get(pos),
                                                   temp_prob) + config_data.laplace_smoothing
                vit_matrix[k, j + 1] = prob
                if prob > max_prob:
                    max_pos = pos
                    max_prob = prob
            backpointer.append([token, max_pos, vit_matrix[:, j + 1].max()])
        return backpointer

    def __compute_transition_probability(self, previous_tag, current_tag):
        """
        method to compute the transition probability from a tag to another
        :param previous_tag: previous tag
        :param current_tag: current tag
        :return: transition probability
        """
        return self.train_manager.count_tags_co_occurrence(previous_tag, current_tag) / \
               self.train_manager.count_pos_tag_frequency(previous_tag)

    def __multiply_probability(self, p1, p2):
        """
        Multiply two probability using logarithm sum property
        :param p1: first probability
        :param p2: second probability
        :return: resulting probability
        """
        if not p1 or not p2:
            return 0.0
        if p1 == 0 or p2 == 0:
            return 0.0
        return math.exp(math.log(p1) + math.log(p2))
