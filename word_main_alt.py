from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

from word_tools import load_doc
import numpy as np

# load doc
in_filename = 'data/aol/full/train.query.sequences.alt2.txt'
# in_filename = 'data/aol/full/temp_check.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

# Trim and keep only length of 5
trim_lines = []
for item in lines:
    if len(item.split()) == 5:
        trim_lines.append(item)

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(trim_lines) # Updates internal vocabulary based on a list of texts.
sequences = tokenizer.texts_to_sequences(trim_lines) # Transforms each text in texts to a sequence of integers.
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# separate into input and output
sequences = array(sequences)
print(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# define model
model = Sequential()
model.add(Embedding(vocab_size, 5, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model batch_size = 128, epochs = 100
model.fit(X, y, batch_size=128, epochs=20)
 
# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))