from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt_tab')

text= [
        "This is the first sentence.",
    "This is the second sentence.",
    "A powerful technique for NLP.",
]


stop_words=set(stopwords.words('english'))

# Tokenzation and removing stop words
sentences=[]
for sentence in text:
    words=word_tokenize(sentence.lower())
    stop_words=set(stopwords.words('english'))
    words=[w for w in words if w not in stop_words]
    sentences.append(words)

print(words)

# Coverting words to vectors
model = Word2Vec(sentences,min_count=1,window=5)

print(model.wv['powerful'])
print(model.wv.most_similar('powerful', topn=5))






