import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt_tab')
nltk.download('stopwords')

text = "Tokenization in NLP is fun! Let's learn how to do it."
stop_words=set(stopwords.words('english'))

words=word_tokenize(text)

filtered_text= [word for word in words if word not in stop_words]
print(filtered_text)


# 2. using Spacy 
import spacy
nlp=spacy.load("en_core_web_sm")

text = "Tokenization in NLP is fun! Let's learn how to do it."

doc=nlp(text)
fil_word=[word.text for word in doc if not word.is_stop]
print(fil_word)






