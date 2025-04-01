from nltk.stem import PorterStemmer, SnowballStemmer

# Initialize stemmers
porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer("english")

# Words to stem
words = ["running", "runner", "runs", "easily", "fairness"]

porter_stems=[porter_stemmer.stem(word) for word in words]
snowball_stems=[snowball_stemmer.stem(word) for word in words]

print("Original Words:", words)
print("Porter Stemming:", porter_stems)
print("Snowball Stemming:", snowball_stems)

# 2. Spacy
import spacy
from nltk.stem import PorterStemmer

nlp=spacy.load("en_core_web_sm")
stemmer=PorterStemmer()
text = "The children are running and their feet are tired from playing better games."

doc=nlp(text)

stems=[stemmer.stem(word.text) for word in doc]
print(stems)