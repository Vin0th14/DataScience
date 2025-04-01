import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer=WordNetLemmatizer()
# Lemmatize words
words=['running','runs','ran','better','feet']
lematized_words=[lemmatizer.lemmatize(word,pos='v') for word in words]
print(lematized_words)

# Default (noun) lemmatization
le_words=[lemmatizer.lemmatize(word) for word in words]
print(le_words)

# 2. using Spacy 
import spacy
nlp=spacy.load("en_core_web_sm")

text = "The children are running and their feet are tired from playing better games."

doc=nlp(text)
lematized_word_spacy=[word.lemma_ for word in doc]
print(lematized_word_spacy)

#NLTK requires specifying the part of speech (e.g., pos='v' for verbs) to achieve accurate results.
#spaCy automatically handles part-of-speech tagging during processing, making it simpler for lemmatization in larger text contexts.

