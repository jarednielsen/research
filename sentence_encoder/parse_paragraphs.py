# parse_paragraphs.py
"""
This file contains quick scripts to preprocess data.
The functions don't return anything, they read from one file and
write to another. Not part of a workflow, but used for data preparation.
We use Numpy for serialization, not pickle.
"""

import numpy as np
import re

def turn_paragraphs_to_sentences(infile, outfile):
    with open(infile, 'r', encoding='utf8') as fp:        
        content = fp.read()
        content = content.replace('“', '"').replace('”', '"').replace("…", "").replace('’', "'").replace('"', '')
        content = content.replace("—", " ")
        # This ignores names such as "Henry B. Eyring". The capturing group is output on the successive line.
        content = re.split(r'([^A-Z]\.)', content)
        # Then we need to join the split lines
        content = [content[2*i]+content[2*i+1] for i in range(len(content)//2)]
        content = [line.replace('\n', ' ') for line in content]
        content = [line.strip() for line in content if len(line.strip()) > 0]
    
    with open(outfile, 'w') as fp:
        fp.write('\n'.join(content))

def create_tsv(sentencefile, tsvfile):
    with open(sentencefile, 'r') as fp:
        content = fp.readlines()

    with open(tsvfile, 'w') as fp:
        fp.write("RandomInt\tSentence\n")
        for line in content:
            num = np.random.randint(0, 2)
            fp.write('{num}\t{sentence}'.format(num=num, sentence=line))

if __name__ == "__main__":
    print("running parse_paragraphs.py")
    # turn_paragraphs_to_sentences(infile="data/bednar_full.txt", outfile="data/bednar_sentences.txt")
    create_tsv(sentencefile="data/bednar_sentences.txt", tsvfile="tensorboard/bednar/bednar_sentences_labels.tsv")