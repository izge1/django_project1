# AI imports
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

#Named entity recognition AI imports
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

# web interface imports
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from .forms import MessageForm
from .models import Message

global currentMessageList # list for section
currentMessageList = ['Hello my name is Chatty!']

# AI variables
words=[]
classes = []
documents = []
ignore_words = ['?', '!']

data_file = open('c:/Users/Garry/django_project/chatbot/static/chatbot/json/intentsFoodShop3.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# AI set-up
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique lemmatized words", words)
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('tf.compat.v1.get_default_graphAI_model.h5', hist)

from keras.models import load_model
model = load_model('tf.compat.v1.get_default_graphAI_model.h5')
model._make_predict_function()
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# AI functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

ERROR_THRESHOLD = 0.25
def predict_class(sentence):
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
   
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# System entity recognition and extraction
def get_sys_entity(msg, intents_json, entity_type):
    msg_u = msg.upper() # convert message to upper case to increase chance of match with entities
    doc = nlp(msg_u)  
    #print(doc)
    entity_type = ""
    entity_value = ""
               
    list_of_entities = intents_json['sys_entities']
        
    for entity in list_of_entities:
        
        entity_values_length = len(list_of_entities)
            
        for count in range(entity_values_length):
            entity_type = entity['entity']['type']
            #print("type: ", entity_type)
            
            #print(doc.ents)
            for ent in doc.ents:
                
                print(ent.label_)
                if ent.label_ == entity_type:
                    #print("Match found! ", ent.text)
                    entity_value = ent.text
                    #print("time: ", entity_value)
                    
    return entity_value
                
# Custom entity values or synonym extraction from user input
def get_entity_parameter(msg, intents_json, entity_name): 
    msg_l = msg.lower() # convert message to lower case to increase chance of match with entities
    entity_n = ""
    entity_parameter_value = ""
    
    list_of_entities = intents_json['entities']
        
    for ents in list_of_entities:
        entity_n = ents['entity']['entity_name']
        
        # If the entity name passed from the calling function is in the json file
        if entity_name == entity_n:
            #print("Match!", entity_n, "\n")
            entity_values = ents['entity']["values"]
            #print("ent vals: ", entity_values)
            entity_values_length = len(entity_values)
            #print("ent vals l: ", entity_values_length)        
            
            # Iterate through the entity values
            
            for count in range(entity_values_length):
                ent_value = ents['entity']["values"][count]['value']
                ent_val_found = ent_value in msg_l
                        
                if ent_val_found:
                    #print ("Ent value found:", ent_value)
                    entity_parameter_value = ent_value
                else:
                    entity_value_synonyms = ents['entity']["values"][count]["synonyms"]
                    ent_val_synonyms_lengh = len(entity_value_synonyms)
                        
                    for count2 in range(ent_val_synonyms_lengh):
                        ent_value_synonym = ents['entity']["values"][count]["synonyms"][count2]
                        ent_val_synonym_found = ent_value_synonym in msg_l
                    
                        # If a synonym was found and value is not empty
                        if ent_val_synonym_found and ent_value_synonym != "":
                            #print("Ent value synonym found:", ent_value_synonym)
                            entity_parameter_value = ent_value
                            
    return entity_parameter_value
                    
# create data structure to store context
context = {}
# set global variables to store pizza data
pType = ""
pSize = ""
pCollection = ""

def getResponse(msg, ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    
    userID = 'izge1' # temp user id for testing
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            
            #check if intent sets a context
            if 'context_set' in i:
                #print ('context: ', i['context_set'])
                context[userID] = i['context_set']
                  
            # check if this intent is contextual and applies to the user's conversation
            if not 'context_filter' in i or \
                 (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                     print ('tag:', i['tag'])
                     # a random response from the intent
                     result = random.choice(i['responses'])
                     
                     # get action if there is one
                     if 'action' in i:
                        action_output = executeAction(i['action'], msg, result, intents_json)
                        result = action_output
            else:
                result = "Sorry I didn't get that. You can do things like ask what pizzas we sell, \ncheck opening times and order pizzas"
                
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg)
    res = getResponse(msg, ints, intents)
           
    return res

# call appropriate action specified by an intent
def executeAction(actionType, msg, result, intents_json):
    output = ""
    if actionType == "order_pizza":
        output = order_pizza(msg, result, intents_json)       
    if actionType == "reset_order":
        output = reset_order(result)
    if actionType == "cancel_order":
        output = cancel_order(result)
    return output

# define functions to implement intent actions

def get_prompt(intents_json, entity_nameP):
    list_of_entities = intents_json['entities']
        
    for ents in list_of_entities:
        entity_n = ents['entity']['entity_name']
        
        # If the entity name passed from the calling function is in the json file
        if entity_nameP == entity_n:
            entity_prompt = ents['entity']["prompt"]
        else:
            list_of_sys_entities = intents_json['sys_entities']
            for ents in list_of_sys_entities:
                entity_ns = ents['entity']['entity_name']
                
                # If the entity name passed from the calling function is a system entity in the json file
                if entity_nameP == entity_ns:
                    entity_prompt = ents['entity']["prompt"]
                    
    return entity_prompt
    
def order_pizza(msg, result, intents_json):
    global pType
    global pSize
    global pCollection
    responseText = result
    
    # Get parameter values from user input and store them in global variables
    pTypeR = get_entity_parameter(msg, intents_json, "pizza_type")
    if pTypeR:
        pType = pTypeR
    pSizeR = get_entity_parameter(msg, intents_json, "pizza_size")
    if pSizeR:
        pSize = pSizeR
    pCollectionR = get_sys_entity(msg, intents_json, "TIME")
    if pCollectionR:
        pCollection = pCollectionR
    
    # If required, get the prompt, otherwise format response to the user and save the order
    if pType == "":
        pizzaOutputString = get_prompt(intents_json, "pizza_type")
    elif pSize == "":
        pizzaOutputString = get_prompt(intents_json, "pizza_size")
    elif pCollection == "":      
        pizzaOutputString = get_prompt(intents_json, "collection_time")
    else:
        pizzaOutputString = responseText.format(pType, pSize, pCollection)
        
        # Code to save order to the database would go here
 
        # Reset the global pizza variables ready for the next order
        pType = ""
        pSize = ""
        pCollection = ""
    
    return pizzaOutputString

def reset_order(result):
    global pType
    global pSize
    global pCollection
    
    pType = ""
    pSize = ""
    pCollection = ""
    
    return result

def cancel_order(result):
    global pType
    global pSize
    global pCollection
    
    pType = ""
    pSize = ""
    pCollection = ""
    
    # Reset context
    global context
    context = {}
    
    return result

# def getResponse(ints, intents_json):
    # tag = ints[0]['intent']
    # list_of_intents = intents_json['intents']
    # for i in list_of_intents:
        # if(i['tag']== tag):
            # result = random.choice(i['responses'])
            # break
    # return result

# def chatbot_response(msg):
    # ints = predict_class(msg)
    # res = getResponse(ints, intents)
    # return res

# view function for the homepage
def index(request):
    return HttpResponse("Hello Django")

# view function for the chat pagr
def message(request):
    if request.method == 'POST':
        f = MessageForm(request.POST)
    
        if f.is_valid():
            # add message to the section
            messageContent = f["messageContent"].value()
            
            currentMessageList.append(messageContent + "\n\n") # add new message to the list
            chatbotResponse = chatbot_response(messageContent) # get chatbot response
            currentMessageList.append(chatbotResponse + "\n\n") # add AI response to the list
            
            length = len(currentMessageList)
            length = int(length)
            if length > 5:
                currentMessageList.pop(0) # remove the first message to prevent scrolling in the section
            
            # add message to the database
           
            f.save()
            r = Message(messageContent=chatbotResponse)
            r.save()
            
            #messages.add_message(request, messages.INFO, 'Message Added.')
            return redirect('message')
    else:
        f = MessageForm()
        messagesL = Message.objects.all() # get records from database
    return render(request, 'chatbot/message.html', {'form': f, 'messagesL': messagesL, 'currentMessageList': currentMessageList})

## view function to display a list of messages
def message_list(request):
    messagesL = Message.objects.all()
    return render(request, 'chatbot/message_list.html', {'messagesL': messagesL})