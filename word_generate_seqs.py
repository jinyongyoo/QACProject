import random
from word_tools import load_doc, clean_line, save_doc

in_filename = 'data/aol/full/train.query.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
# print(doc[:200])

####
# NOTE: We remove terms of size 1 because makes no sense for word prediction
####

print('Filtering, Normalizing, and Creating Sequences out of tokens... ')
sequences = list()
for seq in lines:
    seq_length = seq.split()

    # Keep if more than 1 word in term
    if len(seq_length) > 1:
        tokens = clean_line(seq)
        # print(tokens)
        # convert into a line
        line = ' '.join(tokens)
        # store
        sequences.append(line)
print('Total Sequences: %d' % len(sequences))

print('Saving Sequences to a File')
# save sequences to file
out_file_name = 'data/aol/full/train.query.sequences.txt'
save_doc(sequences, out_file_name)
print('Saved... :)')