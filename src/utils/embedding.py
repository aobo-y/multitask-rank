class EmbeddingMap:
    '''
    convert index to token and vice versa
    '''

    def __init__(self, tokens=[]):
        self.token_2_index = {}
        self.tokens = [t for t in tokens] # clone

        for index, token in enumerate(tokens):
            self.token_2_index[token] = index

    def append(self, token):
        self.tokens.append(token)
        self.token_2_index[token] = len(self.tokens) - 1

    def size(self):
        return len(self.tokens)

    def __contains__(self, token):
        return token in self.token_2_index

    def  __len__(self):
        return len(self.token_2_index)

    def __getitem__(self, key):
        if type(key) is int:
            return self.tokens[key]
        else:
            return self.token_2_index[key]


class Vocabulary(EmbeddingMap):
    '''
    embedding maps for words
    '''

    pad = '<pad>' # Padding token
    sos = '<sos>' # Start of Sentence token
    eos = '<eos>' # End of Sentence token
    unk = '<unk>'  # pretrained word embedding usually has this

    def __init__(self, tokens=[]):
        super().__init__(tokens)

        for special_token in [self.pad, self.sos, self.eos, self.unk]:
            if not special_token in self:
                self.append(special_token)

        self.pad_idx = self[self.pad]
        self.sos_idx = self[self.sos]
        self.eos_idx = self[self.eos]
        self.unk_idx = self[self.unk]

    def words_2_idx(self, words):
        idxes = [
            self[word] if word in self else self.unk_idx
            for word in words
        ]
        # idxes.append(self.eos_idx)
        return idxes


class Persons(EmbeddingMap):
    '''
    embedding maps for persona
    '''

    none = '<none>'

    def __init__(self, tokens=[]):
        super().__init__(tokens)

        self.append(self.none)
        self.none_idx = self[self.none]
