from nlp.utils import remove_non_alpha_numeric_characters, split_into_tokens, count_document_word_frequencies
from nlp.utils import remove_parenthesized_strings, split_into_sentences
from nlp.data.Ted.preprocessing import remove_speaker_names
from nlp.models.gensim_wrapper import train_word2vec
from bokeh.plotting import figure, show
import urllib.request
import numpy as np
import lxml.etree
import zipfile
import os


# Download the dataset if it's not already there: this may take a minute as it is 75MB
if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/"
                               "ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")

# extract the subtitle text of the ted talks from the XML:
with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
input_text = '\n'.join(doc.xpath('//content/text()'))
del doc

i = input_text.find("Hyowon Gweon: See this?")
print(input_text[i-20:i+150])
print()

# remove all parts within parenthezes
input_text = remove_parenthesized_strings(input_text)

i = input_text.find("Hyowon Gweon: See this?")
print(input_text[i-20:i+150])
print()

# split the sentences and store them as members of a list
input_text = split_into_sentences(input_text)

# remove the names of the speakers (heuristically)
text = []
for line in input_text:
    text.append(remove_speaker_names(line))
print(text[0:3])
print()

# remove all characters, which are non alpha numeric
text_temp = []
for x in text:
    x_temp = []
    for sentence in x:
        sentence = remove_non_alpha_numeric_characters(sentence)
        x_temp.append(sentence)
    text_temp.append(x_temp)
text = text_temp
print(text[0:3])
print()

# split the sentences into their tokens
tokenized_text = []
for x in text:
    for sentence in x:
        sentence = split_into_tokens(sentence)
        tokenized_text.append(sentence)
print(tokenized_text[0:3])
print()

# count the word frequencies from the current text
word_frequencies = count_document_word_frequencies(tokenized_text)
for i, key in enumerate(word_frequencies):
    print(key, word_frequencies[key])
    if i==10:
        break
print()

# get the 1000 most frequent words and store them in a list
frequency_list = []
for i, key in enumerate(word_frequencies):
    frequency_list.append(word_frequencies[key])
    if i==999:
        break

# plot a distribution over the 1000 most frequent words
hist, edges = np.histogram(frequency_list, density=True, bins=1000)
p = figure(tools="pan,wheel_zoom,reset,save", toolbar_location="above", title="Top-1000 words distribution")
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="#555555")
show(p)

# train word embeddings on the ted corpus
model_ted = train_word2vec([tokenized_text])

# get the most similar word embeddings for different categories
print()
print(model_ted.wv.most_similar("man"))
print()
print(model_ted.wv.most_similar("computer"))
print()

# get the first 100 words as strings
words_top_ted = []
for key in word_frequencies:
    words_top_ted.append(key)

# get the corresponding word embeddings from the trained word2vec model
words_top_vec_ted = model_ted.wv[words_top_ted]
print(type(words_top_vec_ted))
