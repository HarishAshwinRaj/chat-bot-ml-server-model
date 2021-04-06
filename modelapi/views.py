from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
# Create your views here.
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json


'''
remeber to send the request in json information

'''
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
import tensorflow
import random
import json
import io
import os
def model_assembler(x_size,y_size):
    in_size =x_size
    out_size = y_size
    word_size =46
    net = tflearn.input_data(shape=[None,in_size])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, out_size, activation="softmax")
    net = tflearn.regression(net)
    model = tflearn.DNN(net)
    return model
#model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)

def loadfiles():

    s= os.path.abspath(__file__)
    print(type(s))
    s,j = os.path.split(s)
    print(s)
    with open(s+'/mlmodel/intents.json') as i:
        data = json.load(i)
    with open(s+'/mlmodel/words.json') as f:
        w = json.load(f)
    words = w
    with open(s+'/mlmodel/labels.json') as g:
        datas = json.load(g)
    labels = datas
    model = model_assembler(len(words),len(labels))
    model.load(s+"/mlmodel/model.tflearn")
    return words,labels,data,model
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

words =False
labels =False
data = False
model = False
def chat(question):
    print("Start talking with the bot (type quit to stop)!")
    inp = question
    ''' loading the model'''
    global model
    if( model == False):
        print("------loading_model--------")
        global words
        global labels
        global data
        words,labels,data,model = loadfiles()

    results = model.predict([bag_of_words(inp,words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    if results[results_index]>0.9:
      for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
      f = random.choice(responses)
    else:
      f = "I didn't understand that."
    print("-----api-----chat -----bot-------")
    print(f)
    return f


''' ------ api ----part------'''

@csrf_exempt
@require_http_methods([ "POST"])
def posthandler(request):
    print ("helll")
    received_json_data = json.loads(request.body.decode("utf-8"))
    print(received_json_data['question'])
    answer = chat(received_json_data["question"])
    resp = {'answer':answer}
    return JsonResponse(resp)
