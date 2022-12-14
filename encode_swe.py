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
        self.content_repatter1 = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)")
        self.content_repatter2 = re.compile(r"[A-Za-z0-9\._+]*@[\-_0-9A-Za-z]+(\.[A-Za-z]+)*")
        self.content_repatter3 = re.compile(r'[\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}')
        self.content_repatter4 = re.compile(r"([12]\d{3}[/\-年])*(0?[1-9]|1[0-2])[/\-月]((0?[1-9]|[12][0-9]|3[01])日?)*(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*")
        self.content_repatter5 = re.compile(r"(明治|大正|昭和|平成|令和|㍾|㍽|㍼|㍻|\u32ff)\d{1,2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*")
        self.content_repatter6 = re.compile(r'((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*億)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*万)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*千)*(0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*(千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+(\(税込\)|\(税抜\)|\+tax)*')
        keisen = "─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿"
        blocks = "▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞▟"
        self.content_trans1 = str.maketrans({k:'<BLOCK>' for k in keisen+blocks})

    def __len__(self):
        # Return the number of tokens
        return len(self.bpe)

    def clean_text(self, text):
        text = self.content_repatter1.sub('<URL>', text)
        text = self.content_repatter2.sub('<EMAIL>', text)
        text = self.content_repatter3.sub('<TEL>', text)
        text = self.content_repatter4.sub('<DATE>', text)
        text = self.content_repatter5.sub('<DATE>', text)
        text = self.content_repatter6.sub('<PRICE>', text)
        while '<BLOCK><BLOCK>' in text:
            text = text.replace('<BLOCK><BLOCK>', '<BLOCK>')
        return text

    def encode(self, text, clean=False):
        '''
        Find the longest matching word and create a sequence of tokens
        with tokens with that word written by the list
        '''
        # Text preprocessing
        text = text.replace(' ', '<SP>')
        text = text.replace('  ', '<SP>')
        text = text.replace('\r\n', '<BR>')
        text = text.replace('\n', '<BR>')
        text = text.replace('\r', '<BR>')
        text = text.replace('\t', '<TAB>')
        text = text.replace('—', 'ー')
        text = text.replace('−', 'ー')
        for k,v in self.emoji['emoji'].items():
            if k in text:
                text = text.replace(k, v)
        if clean:
            text = text = self.clean_text(text)

        def check_synbol(x):
            '''Function to check if the character is a symbol'''
            e = x.encode()
            if len(x) == 1 and len(e) == 1:
                c = (int(e[0])<<8)+int(e[1])
                if (c >=0xc2a1 and c <= 0xc2bf) or \
                    (c >= 0xc780 and c <= 0xc783) or \
                    (c >= 0xcab9 and c <= 0xcbbf) or \
                    (c >= 0xcc80 and c <= 0xcba2):
                        return True
            return False

        def check_u2e(x):
            '''Function to check if the character is a 3-byte symbol'''
            e = x.encode()
            if len(x) == 1 and len(e) == 3:
                # Create character code
                c = (int(e[0])<<16)+(int(e[1])<<8)+int(e[2])
                if c >= 0xe28080 and c <= 0xe2b07f:
                    return True
            return False

        pos = 0
        result = []
        while pos < len(text):
            '''Until the end of the text'''
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

    def decode(self, tokens, breakline='\n'):
        '''Convert tokens to text'''
        words = []
        byte_tokens = []
        for i in tokens:
            # Obtain character string representation from token list
            word = self.bpe[i][0]
        if word[:6] == '<|byte' and word[-2:] == '|>':
            # Encoded in a byte sequence
            byte_tokens.append(int(word[6:-2]))
        else:
            # If put the byte sequence back together
            if len(byte_tokens) > 0:
                words.append(bytearray(byte_tokens).decode('utf-8' , errors='replace'))
                byte_tokens = []
            # Undo string from token
            if word[:7] == '<|emoji' and word[-2:] == '|>':
                # Emoji
                words.append(self.emoji['emoji_inv'][word])
            elif word == '<BR>':
                # Line break
                words.append(breakline)
            elif word == '<SP>':
                # Space
                words.append(' ')
            elif word == '<TAB>':
                # Tab
                words.append('\t')
            elif word == '<BLOCK>':
                #  Block
                words.append('▀')
            elif word == '<KIGOU>':
                # Kigou
                words.append('ǀ')
            elif word == '<U2000U2BFF>':
                # 3-byte symbol
                words.append('‖')
            else:
                words.append(word)
        if len(byte_tokens) > 0:
            # Undo the remaining byte sequence
            words.append(bytearray(byte_tokens).decode('utf-8' , errors='replace'))
        text = ''.join(words)
        return text

if __name__ == '__main__':
    import argparse
    import shutil
    import os
    import json
    from tqdm import tqdm
    import pickle
    import uuid
    from multiprocessing import Pool

    # Options
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', help='source dir', required=True)
    parser.add_argument('--dst_dir', help='destnation file', required=True)
    parser.add_argument('--tmp_dir', help='tempolary file', default='tmpfiles')
    parser.add_argument('--vocaburaly', help='vocaburaly file', default='ja-swe32k.txt')
    parser.add_argument('--num_process', help='process num', type=int, default=8)
    parser.add_argument('--combine', help='Concatenate files with <|endoftext|> separator into chunks of this minumum size', type=int, default=50000)
    parser.add_argument('--clean_text', action='store_true')
    parser.add_argument('--tmpsilze', help='num chunks in tempolary file', type=int, default=5000)
    args = parser.parser.parse_args()

    # Create tempolary directory
    if os.path.isdir(args.tmp_dir):
        shutil.rmtree(args.tmp_dir)
    os.mkdir(args.tmp_dir)

    # Create an encoder
    with open(args.vocaburaly, encoding='utf-8') as f:
        bpe = f.read().split('\n')
    with open('emoji.json', encoding='utf-8') as f:
        emoji = json.loads(f.read())
    enc = SWEEncoder_ja(bpe, emoji)

    # Functions executed in a separate process
    array_file = []
    def _proc(i):
        token_chunks = []
        raw_text = ''

        # Process only corresponding files
        for j, (curDir, dirs, files) in enumerate(array_file):
            if not (j % args.num_process == i):
                continue
            print('append #', curDir)
            for file in tqdm(files):
                if file.endswith('.txt'):
                    # Encode file
                    input = os.path.join(curDir, file)
                    with open(input, encoding='utf-8') as f:
                        raw_text += f.read()
                    raw_text += '<|endoftext|>'
                    # Connecting a sequence of encoded tokens
                    if len(raw_text) >= args.combine:
                        tokens = np.stack(enc.encode(raw_text, clean=args.clean_text))
                        token_chunks.append(tokens)
                        raw_text = ''
            if raw_text and len(raw_text) > 0:
                tokens = np.stack(enc.encode(raw_text))
                token_chunks.append(tokens)
            # When encoded data in memory becomes large, write it out
            if len(token_chunks) > args.tmpslize:
                with open(os.path.join(args.tmp_dir, '%s.pkl'%str(uuid.uuid())), 'wb') as f:
                    pickle.dump(token_chunks, f)
                    token_chunks = []

        # Exporting encoded data
        with open(os.path.join(args.tmp_dir, '%s.pkl'%str(uuid.uuid())), 'wb') as f:
            pickle.dump(token_chunks, f)

        # Make a list of directories containing files to be processed
        for curDir, dirs, files in os.walk(args.src_dir):
            array_file.append((curDir, dirs, files))

        # Execute in parallel
        with Pool(args.num_process) as p:
            p.map(_proc, list(range(args.num_process)))

        # Read and connect all resulting files
        token_chunks = []
        for s in os.listdir(args.tmp_dir):
            with open(os.path.join(args.tmp_dir, s), 'rb') as f:
                token_chunks.extend(pickle.load(f))

        # Save the result
        np.savez_compressed(args.dst_file, *token_chunks)
        shutil.rmtree(args.tmp_dir)


