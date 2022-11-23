import re

import numpy as np


class SWEEncoder_ja:
    '''Encoding Algorithm'''
    def __init__(self, bpe, emoji):
        '''Initialize the class'''
        self.bpe = [[b] if (b==',' or ',' not in b) else b.split(',') for b in bpe]
        self.swe = {}
        for idx, b in enumerate(self.bpe):
            for wb in b:
                self.swe[wb] = idx
        self.emoji = emoji
        self.maxlen = np.max([len(w) for w in self.swe.keys()])

    def __len__(self):
        # Return the number of tokens
        return len(self.bpe)

    def encode(self, text, clean=False):
        '''
        Find the longest matching word and create a sequence of tokens
        with tokens with that word written by the list
        '''
        # Clean text
        def check_synbol(x):
            '''Function to check if the character is a symbol'''

        def check_u2e(x):
            '''Function to check if the character is a 3-byte symbol'''

        pos = 0
        result = []
        while pos < len(text):
            '''until the end of the text'''
            # Viewing position in the text
            end = min(len(text), pos+self.maxlen+1) if text[pos] == '<' else pos+3
            # Matching candidate tokens
            candidates = []
            for e in range(end, pos, -1):
                '''Find matching tokens'''
                # Check if the token is in the dictionary
                wd = text[pos:e]
                if wd in self.swe:
                    if wd[0] == '<' and len(wd) < 2:
                        candidates = [(self.swe[wd], e)]
                        break
                    else:
                        candidates.append((self.swe[wd], e))
            if len(candidates) > 0:
                wp, e = sorted(candidates, key=lambda x: x[0])[0]
                result.append(wp)
                pos = e
            # If not on the token list
            else:
                end = pos + 1
                wd = text[pos:end]
                if check_synbol(wd):
                    result.append(self.swe['<SYMBOL>'])
                elif check_u2e(wd):
                    result.append(self.swe['<U2000U2BFF>'])
                else:
                    # Encode to byte string
                    for i in wd.encode('utf-8'):
                        # Encoding in <|byte0|> ~ <|byte255|> columns
                        result.append(self.swe['<|byte%d|>'%i])
                    pos = end
        return result



