"""
Regole da inglese a italiano:
- Eng: aggettivo -> nome -- Ita: nome -> aggettivo (inversione)
- Eng: are -> not -- Ita: non -> sono (inversione)
- Eng Nome's -- Ita: di Nome
"""
dictionaries = {
    'd1': {'The': 'Il', 'black': 'nero', 'droid': 'droide', 'then': 'quindi', 'lowers': 'abbassa', "Vader": 'Vader',
           "'s": 'di', 'mask': 'maschera', 'and': 'e', 'helmet': "elmo", 'onto': 'sulla', 'his': 'sua',
           'head': 'testa'},
    'd2': {'These': 'questi', 'are': 'sono', 'not': 'non', 'the': 'i', 'droids': 'droidi',
           'you': 'tu', "'re": 'stai', 'looking for': 'cercando'},
    'd3': {'Your': 'tuoi', 'friends': 'amici',
           'may': 'potrebbero', 'escape': 'scappare', ',': ',', 'but': 'ma', 'you': 'tu', 'are': 'sei',
           'doomed': 'condannato'},
    'd4': {'Help me': 'Aiutami', ',': ',', 'Obi-Wan-Kenobi': 'Obi-Wan-Kenobi', '.': '.', 'You': 'Tu',
           "'re": 'sei', 'my': 'mia', 'only': 'sola', 'hope': 'speranza'},
    'd5': {'I': 'Io', 'find': 'trovo', 'your': 'tua', 'lack': 'mancanza', 'of': 'di', 'faith': 'fede',
           'disturbing': 'insopportabile'},
    'd6': {'No': 'No', '.': '.', 'I': 'Io', 'am': 'sono', 'your': 'tuo', 'father': 'padre'},
    'd7': {'Now': 'Adesso', ',': ',', 'young': 'giovane', 'Skywalker': 'Skywalker',
           'you': 'tu', 'will die': 'morirai'},
    'd8': {'Fear': 'Paura', 'is': 'è', 'the': 'il', 'path': 'cammino', 'to': 'per', 'dark': 'oscuro', 'side': 'lato'}
}


def translate_sentence(tagged_sentence, dict_name):
    translation = []
    tagged_sentence_length = len(tagged_sentence)
    dictionary = dictionaries.get(dict_name)

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
        translation.append(dictionary.get(e[0]))

    return translation
