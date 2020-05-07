import random
from word_tools import load_doc, clean_line, save_doc
# GPU
# import keras
# import tensorflow as tf


# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)

in_filename = 'data/aol/full/test.query.txt'
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

    # Keep ONLY sequences of length 5 
    if len(seq_length) == 5:
        tokens = clean_line(seq)
        # convert into a line
        line = ''.join(tokens)
        # store
        sequences.append(line)

    # # Shorten sequences of length 5
    # if len(seq_length) >= 5:
    #     tokens = clean_line(seq)
    #     line = ' '
    #     iter = 0
    #     for tok in tokens:
    #         if iter >= 5:
    #             break
    #         line = line.join(tok)
    #         iter += 1
    #     sequences.append(line)


print('Total Sequences: %d' % len(sequences))

print('Saving Sequences to a File')
# save sequences to file
out_file_name = 'data/aol/full/train.query.sequences.alt2.test.txt'
save_doc(sequences, out_file_name)
print('Saved... :)')