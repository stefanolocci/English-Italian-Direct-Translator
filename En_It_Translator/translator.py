dictionary = {'the': 'i', 'black': 'nero', 'droid': 'droide', 'then': 'quindi', 'lowers': 'abbassa', "Vader's": 'Vader',
              'mask': 'maschera', 'and': 'e', 'helmet': 'elmo', 'onto': 'sulla', 'his': 'sua', 'head': 'testa',
              'these': 'questi', 'are': 'sei', 'not': 'non', 'droids': 'droidi', 'you': 'tu', "'re": 'stai',
              'looking': 'cercando', 'for': 'per', 'your': 'tuoi', 'friends': 'amici', 'may': 'potrebbero',
              'escape': 'scappare', ',': ',', 'but': 'ma', 'doomed': 'condannato'}


def translate_word(word):
    return dictionary.get(word)


def translate_sentence(sentence):
    translation = []
    for word in sentence:
        translation.append(translate_word(word))
    return translation
