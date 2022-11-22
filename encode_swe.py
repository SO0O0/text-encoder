import re

import numpy as np

class SWEEncoder_ja:
    '''Encoding Algorithm'''
    def __init__(self, bpe, emoji):
        self.bpe = [[b] if (b==',' or ',' not in b) else b.split(',') for b in bpe]
        self.swe = {}
        for idx, b in enumerate(self.bpe):
            for wb in b:
                self.swe[wb] = idx
        self.emoji = emoji
        self.maxlen = np.max([len(w) for w in self.swe.keys()])

    def __len__(self):
        return len(self.bpe)
