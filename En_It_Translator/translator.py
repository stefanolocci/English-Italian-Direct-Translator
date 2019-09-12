"""
Regole da inglese a italiano:
- Eng: aggettivo -> nome -- Ita: nome -> aggettivo (inversione)
- Eng: are -> not -- Ita: non -> sono (inversione)
- Eng Nome's -- Ita: di Nome


"""

dictionary = {'The': 'Il', 'black': 'nero', 'droid': 'droide', 'then': 'quindi', 'lowers': 'abbassa', "Vader": 'Vader',
              "'s": 'di', 'mask': 'maschera', 'and': 'e', 'helmet': "elmo", 'onto': 'sulla', 'his': 'sua',
              'head': 'testa', 'These': 'questi', 'are': 'sono', 'not': 'non', 'the': 'i', 'droids': 'droidi',
              'you': 'tu', "'re": 'stai', 'looking for': 'cercando', 'Your': 'tuoi', 'friends': 'amici',
              'may': 'potrebbero', 'escape': 'scappare', ',': ',', 'but': 'ma', 'doomed': 'condannato'}


def translate_word(word):
    return dictionary.get(word)


def translate_sentence(tagged_sentence):
    translation = []
    tagged_sentence_length = len(tagged_sentence)
    for i, tagged_elem in enumerate(tagged_sentence):
        word, tag, prob = tagged_elem
        if i < tagged_sentence_length - 1:
            next_word, next_tag, next_prob = tagged_sentence[i + 1]
            if tag == 'ADJ' and next_tag == 'NOUN':
                tagged_sentence[i], tagged_sentence[i + 1] = tagged_sentence[i + 1], tagged_sentence[i]
            elif (tag == 'VERB' or tag == 'AUX') and next_word == 'not':
                tagged_sentence[i], tagged_sentence[i + 1] = tagged_sentence[i + 1], tagged_sentence[i]
            if next_word == "'s":
                tagged_sentence[i], tagged_sentence[i + 2] = tagged_sentence[i + 2], tagged_sentence[i]
    for e in tagged_sentence:
        translation.append(translate_word(e[0]))

    return translation
