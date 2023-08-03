import nltk  # need to download punkt  nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from nltk.tokenize import word_tokenize

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

#we load the data from file into variable data
with open("intents.json") as file:
    data = json.load(file)

# this try except is so we dont have to run the whole code everytime we run the model, 
# so we are going to save the info and if its not existent we are going to do the except
# being the whole process
try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)
        print("datapickle")
except:
    # create lists we will need
    words = [] # list for words that appear in the file
    labels = [] # tags in json file
    docs_x = [] # words in the pattern
    docs_y = [] # tag that the docsx word belong to

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = word_tokenize(pattern) 
            words.extend(wrds) # we add the wrds to the words list, extend adds it to the end of the list
            docs_x.append(wrds) # pattern added to the list
            docs_y.append(intent["tag"]) # what tag its a part of, so each entry has a pattern in docsx and tag in docsy

        if intent["tag"] not in labels: # adds all the tags in the labels list
            labels.append(intent["tag"])

    # a stemmer is a algorithm that just leaves the root of the words search lancaste stemmer for more info
    words = [stemmer.stem(w.lower()) for w in words if w != "?"] 
    # removes all duplicated words
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []
    out_empty= [0 for _ in range(len(labels))]

    # here we loop in the patterns in the docs_x
    for x, doc in enumerate(docs_x):# x is the number  from the enumeration and doc = word
        bag = [] # bag of words, one hot encoded
        wrds = [stemmer.stem(w) for w in doc] #we stem again because we only stemmed in words after we added them to wrds


        # this is the one hot encoding, so if the word appears is in the current pattern its a 1 if not its a 0
        # what happens here is that its going pattern by pattern in the docs_x, so word by word in the words list,
        #  if the word its in the wrds list, the wrds list being the wrds that appear in a specific pattern, 
        # it will one hot encode it in the bag of words
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        #copy of output_empty
        output_row = out_empty[:]
        # x in docs_y, we index it and change the 0 to a 1 
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    # array() makes the list of lists into multidimensional arrays so we can fit them into the model
    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
    print("primer except funciona")


#tensorflow.reset_default_graph() is out of date so  im gonna use 
tensorflow.compat.v1.reset_default_graph()

# for our input layer the is n, n = bag of words
# then we have 2 hidden layers with 8 neurons, each input connect to each layer, and the 8 neurons in the first layer 
# connect to the next 8 neurons from the next layer 
# our output layer will have as many neurons as labels we have, it has softmax activation that gives probability 
# to each neuron, so it works with the higher probability so throw a response from the higher probability

net = tflearn.input_data(shape=[None, len(training[0])])#this defines the input shape expected is from training 0 length
net = tflearn.fully_connected(net,8) # 8 neurons for this hidden layer
net = tflearn.fully_connected(net,8) # 8 neurons for this hidden layer
net = tflearn.fully_connected(net,len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


# this try is for loading a model if theres one already created and we dont train it again, try and except didnt work
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size= 8, show_metric=True)
    model.save("model.tflearn")



# if we add or change information from json file we have to delete existing model 
# and pickle file

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]

    s_words = word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se: # If current word in sentence its equal to a word in the list words
                bag[i] = (1)
    return numpy.array(bag)

def chat():
    print("The bot is ready!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        results = model.predict([bag_of_words(user_input, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.9:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("What you on?")
        
        
chat()
        
# next step is to implement the bot in discord and add more intents 