# 1. NLTK

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt_tab')

text = "Tokenization in NLP is fun! Let's learn how to do it."

word_tokens=word_tokenize(text)
print(word_tokens)
sent_tokens=sent_tokenize(text)
print(sent_tokens)

# 2. using Hugging face

from transformers import AutoTokenizer
# Load a pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Tokenization in NLP is fun! Let's learn how to do it."

# Subword-level tokenization
tokens=tokenizer.tokenize(text)
print(tokens)

# Token IDs
token_id=tokenizer.convert_tokens_to_ids(tokens)
print(token_id)

# 3. using Keras
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
from keras.src.legacy.preprocessing.text import Tokenizer,text_to_word_sequence

tokenizer=Tokenizer()
tokenizer.fit_on_texts(text)
tokens=text_to_word_sequence(text)
print(tokens)


