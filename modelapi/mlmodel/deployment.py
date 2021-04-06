import nltk
#nltk.download('punkt')
""" check wether punkt is already present and trying to download if not present"""
try:
    print("resourse punkt found ")
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("punkt is downloading")
    nltk.download('punkt')


from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()


import numpy
import tflearn
#import tensorflow
import random
import json
import io


in_size =46
out_size = 6
word_size =46
net = tflearn.input_data(shape=[None,in_size])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, out_size, activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
with open('intents.json') as i:
    data = json.load(i)
model.load("model.tflearn")

# list of words are required so please do it nexttime remainder

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)
with open('intents.json') as i:
    data = json.load(i)
with open('words.json') as f:
    w = json.load(f)


words = w
with open('labels.json') as g:
    datas = json.load(g)
labels = datas
def chat(question,words,labels):
    print("Start talking with the bot (type quit to stop)!")
    inp = question
    if inp.lower() == "quit":
        pass

    results = model.predict([bag_of_words(inp,words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    if results[results_index]>0.7:
      for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
      print(random.choice(responses))
    else:
      print("I didn't understand that.")
