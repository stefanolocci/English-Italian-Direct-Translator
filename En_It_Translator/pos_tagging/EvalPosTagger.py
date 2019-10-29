from pos_tagging.CorpusManager import CorpusManager
from utils import config_data


class EvalPosTagger:
    def __init__(self, test_manager):
        self.test_tags = test_manager.tag_list

    def compute_accuracy(self, predicted_tags):
        counter = 0
        for i, pt in enumerate(predicted_tags):
            if pt[1] != self.test_tags[i]:
                # print("{}, {}, {}".format(pt[0], pt[1], real_tags[i]))
                counter += 1
        return (len(self.test_tags) - counter) / len(self.test_tags)
