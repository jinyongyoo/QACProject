# Load Document into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# # load document
# in_filename = 'republic_clean.txt'
# doc = load_doc(in_filename)
# lines = doc.split('\n')
# print(doc[:200])

import string

def getfile(filename,results):
   f = open(filename)
   filecontents = f.readlines()
   for line in filecontents:
     foo = line.strip('\n')
     results.append(foo)
   return results

# clean a line
def clean_line(line):
	# replace '--' with a space ' '
	line = line.replace('--', ' ')
	# split into tokens by white space
	tokens = line.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens

# turn a doc into clean tokens
def clean_doc(doc):
	# replace '--' with a space ' '
	doc = doc.replace('--', ' ')
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens

# # clean document
# tokens = clean_doc(doc)
# print(tokens[:200])
# print('Total Tokens: %d' % len(tokens))
# print('Unique Tokens: %d' % len(set(tokens)))

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()