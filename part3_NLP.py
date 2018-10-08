
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Text formatting ignoring non english words
stop = stopwords.words('english')
stop.append("n't")
# Opening and reading file
f = open('potter.txt')
raw = f.read()
# Tokenizing words in the file
tokens = nltk.word_tokenize(raw)
# Word formatting, ignores words with less than 3 characters
tokens = [token for token in tokens if token not in stop]
tokens = [word for word in tokens if len(word) >= 3]#
tokens = [word.lower() for word in tokens]
# Part a: Lemmatization
lemmatizer=WordNetLemmatizer()# Creating an instance of WordNetLemmatizer
print("lemmatized words:")
for word in tokens:
    print(word)#print whe words in file
    print(lemmatizer.lemmatize(word,'v'))#prints the base form of verbs in words file
# Create bigrams
bgs = nltk.bigrams(tokens)

# compute frequency distribution for all the bigrams in the text
fdist = nltk.FreqDist(bgs)
# Selecting and printing the top 5 bigrams
top_grams2 = sorted(fdist.items(), key=lambda x: x[1], reverse=True)[:5]
print(top_grams2)

# sent_text = nltk.sent_tokenize(raw)
tg = zip(*(top_grams2))
tg = list(tg)
bg1 = tg[0]
print(bg1)

# Creating tokens of sentences in input file
sent_text = nltk.sent_tokenize(raw)
sent_text = [word.lower() for word in sent_text]

summary = [] # array hold sentences that have the top 5 most repeated bigrams
for i in range(5):
    for sent in sent_text:
        if bg1[i][0] and bg1[i][1] in word_tokenize(sent):
            summary.append(sent)
    # my_sentence=[sent for sent in sent_text if bg1[i][0] and bg1[i][1] in word_tokenize(sent)]
print(*summary)
