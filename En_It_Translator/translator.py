"""
Regole da inglese a italiano:
- Eng: aggettivo -> nome -- Ita: nome -> aggettivo (inversione)
- Eng: are -> not -- Ita: non -> sono (inversione)
- Eng Nome's -- Ita: di Nome
"""


class Translator:

    def __init__(self):
        self.dictionaries = {
            'd1': {'The': 'Il', 'black': 'nero', 'droid': 'droide', 'then': 'quindi', 'lowers': 'abbassa',
                   "Vader": 'Vader',
                   "'s": 'di', 'mask': 'la maschera', 'and': 'e', 'helmet': "l'elmo", 'onto': 'sulla', 'his': 'sua',
                   'head': 'testa'},
            'd2': {'These': 'Questi', 'are': 'sono', 'not': 'non', 'the': 'i', 'droids': 'droidi',
                   'you': 'che tu', "'re": 'stai', 'looking': 'cercando', 'for': ''},
            'd3': {'Your': 'I tuoi', 'friends': 'amici',
                   'may': 'potrebbero', 'escape': 'scappare', ',': ',', 'but': 'ma', 'you': 'tu', 'are': 'sei',
                   'doomed': 'condannato'},
            'd4': {'I': 'Io', 'find': 'trovo', 'your': 'la tua', 'lack': 'mancanza', 'of': 'di', 'faith': 'fede',
                   'disturbing': 'insopportabile'},
            'd5': {'Fear': 'La paura', 'is': 'è', 'the': 'il', 'path': 'cammino', 'to': 'per', 'dark': 'oscuro',
                   'side': 'lato'},
            'd6': {'She': 'Lei', 'wear': 'indossa', 'a': 'un', 'beautiful': 'bellissimo', 'dress': 'vestito'},
            'd7': {'Mary': 'Mary', 'has': 'ha', 'a': 'una', 'collection': 'collezione', 'of': 'di',
                   'expensive': 'costosi',
                   'jewels': 'gioielli'},
            'd8': {'Do': ' ', 'you': 'ti', 'like': 'piacciono', 'fresh': 'fresche', 'cakes': 'le torte',
                   'with': 'con', 'sweet': 'dolce', 'cream': 'la panna', '?': '?'}
        }

    def __preprocess_sentence(self, sentence):
        """
        Preprocess the sentence in order to simplify the translation
        :param sentence: Sentence to preprocess
        :return: processed sentence
        """
        for i, elem in enumerate(sentence):
            word, tag = elem[0], elem[1]
            if i < len(sentence[:-1]):
                next_word, next_tag = sentence[i + 1][0], sentence[i + 1][1]
                if word.lower() == 'do' and next_tag == 'PRON':
                    sentence.pop(i)
        return sentence

    def translate_sentence(self, tagged_sentence, dict_name):
        """
        Direct translate from english to italian and reorder a given sentence using a dictionary
        :param tagged_sentence: Sentence to translate
        :param dict_name: sentence reference dictionary
        :return: transalted and reordered sentence
        """
        translation = ""
        tagged_sentence = self.__preprocess_sentence(tagged_sentence)
        tagged_sentence_length = len(tagged_sentence)
        dictionary = self.dictionaries.get(dict_name)
        for i, tagged_elem in enumerate(tagged_sentence):
            word, tag, prob = tagged_elem
            if i < tagged_sentence_length - 1:
                next_word, next_tag, next_prob = tagged_sentence[i + 1]
                if tag == 'ADJ' and next_tag == 'NOUN':
                    tagged_sentence[i], tagged_sentence[i + 1] = tagged_sentence[i + 1], tagged_sentence[i]
                elif (tag == 'VERB' or tag == 'AUX') and next_word == 'not':
                    tagged_sentence[i], tagged_sentence[i + 1] = tagged_sentence[i + 1], tagged_sentence[i]
                elif tag == 'PART' and word == "'s":
                    tagged_sentence[i], tagged_sentence[i + 1] = tagged_sentence[i + 1], tagged_sentence[i]
                    tagged_sentence[i - 1], tagged_sentence[i] = tagged_sentence[i], tagged_sentence[i - 1]
                    tagged_sentence[i], tagged_sentence[i + 1] = tagged_sentence[i + 1], tagged_sentence[i]
        for e in tagged_sentence:
            translation += dictionary.get(e[0]) + " "
        return translation
