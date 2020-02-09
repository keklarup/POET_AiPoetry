### Functions for using P.O.E.T to generate poems.

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
from keras.callbacks import ModelCheckpoint
import random
import os


def corpusPreparation(data):
    """
    Prep corpus data.
    Each poem in collected corpus seperated from next poem by 4 returns ('\n').
    Replacing those returns with end/start tokens.
    Replacing other returns with 'returntoken' because line ends so important in poems.
    """
    data=data.lower().replace('\n\n\n\n','<endtoken>\n<starttoken>')
    data=data.replace('\n',' <returntoken> ')
    return data
    
def dataset_preparation(data, num_words=None):
    """Prep the corpus text for training.
    Expect end tokens, start tokens, return tokens to already have been added.
    data--corpus of text
    num_words--max number of words for model to have.
    """
    tokenizer = Tokenizer()
    #Want to have system have way to end a poem. So adding another end token.
    corpus = data.lower().replace('<endtoken>','<endtoken2><endtoken>').split("<endtoken>")
    tokenizer.fit_on_texts(corpus)
    ###Generate list of words:
    #If max number of words given, find and use just those words:
    if num_words!=None:
        tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= num_words}
        #make sure to still have out of vocabulary token:
        tokenizer.word_index[tokenizer.oov_token] = num_words + 1
    total_words = len(tokenizer.word_index) + 1
    ###Generate list of input sequences
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,   
                          maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return tokenizer, predictors, label, max_sequence_len, total_words

def initialize_model(predictors, label, max_sequence_len, 
                 total_words, device='/cpu:0'):
    """Sets the parameters for the model.
    Even if loading pretrained model, need to initialize.
    """
    print(device)
    input_len = max_sequence_len - 1
    with tf.device(device):
        model = Sequential()
        model.add(Embedding(total_words, 10, input_length=input_len))
        #model.add(LSTM(250,return_sequences=True))
        #model.add(Dropout(0.2))
        model.add(LSTM(600))
        model.add(Dropout(0.2))
        model.add(Dense(total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model
    
def train_model(model, predictors, label,num_epochs=2, batch_size=100, device='/cpu:0',
                model_name=None, save=False):
    """
    Train a previously defined model
    """
    with tf.device(device):
        if save==True and model_name != None:
            # define the checkpoint
            filepath = model_name
            checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                                             save_best_only=True, mode='min')
            #callbacks_list = [checkpoint]
            callbacks_list = [EarlyStopping(monitor='loss', patience=2),checkpoint]
            try:
                model.load_weights(model_name)
                print('found previous model. Loading weights.')
            except:
                print('No previous model found.')
            model.fit(predictors, label, epochs=num_epochs, verbose=1,
                      callbacks=callbacks_list,batch_size=batch_size)
        else:
            model.fit(predictors, label, epochs=num_epochs, verbose=1,
                     batch_size=batch_size)
    return model
    
def load_model(model, filename):
    """
    Load a previously defined model
    """
    try:
        # define the checkpoint
        filepath =filename
        #checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        #callbacks_list = [EarlyStopping(monitor='loss', patience=2),checkpoint]
        #callbacks_list = [checkpoint]
        model.load_weights(filename)
        print('found previous model. Loading weights.')
    except:
        print('No previous model found.')
    return model    

def generate_text(model, tokenizer, seed_text, max_sequence_len, total_words=10):
    """Generates text using the most likely next token.
    Currently has a tendency to plagarize from the training data. So not adding this option to the notebook.
    """
    for j in range(total_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen= 
                             max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
  
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        if output_word=='endtoken2':
            print('End token found. Stopping early.')
            break
        seed_text += " " + output_word
    return seed_text

def generate_text_random(model, tokenizer, total_words, max_sequence_len, seed_text="starttoken", top_n=10):
    """Generates text using the (slightly modified) probabilities from the model.
    Probabilities are modified to insure total sums to 1 (issues with machine precision)
    """
    word_dict=tokenizer.word_index
    word_dict2 = {value:key for key, value in word_dict.items()} 
    for i in range(total_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], 
                                   maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        #limiting prediction options to just the top n for machine rounding issues.
        predictions=pd.Series(model.predict(token_list, verbose=0)[0]).sort_values(ascending=False)[0:top_n]
        predictions2=(predictions/predictions.sum())
        #rounding for machine precision issues
        predictions2=predictions2.apply(lambda x: np.round(x,2))
        #error is what remains after rounding. Adding to top choice.
        error=1-predictions2.sum()
        predictions2[predictions2.index[0]]=predictions2[predictions2.index[0]]+error
        #random selection of next word using probs from distribution.
        C=np.random.choice(predictions2.index, p=predictions2)
        output_word=word_dict2[C]
        if output_word=='endtoken2':
            print('End token found. Stopping early.')
            break
        seed_text += " " + output_word
    return seed_text



def augmented_writing(seed_text, model, tokenizer,max_sequence_len, top_n=10):
    """
    Allows the user to select next word. Also provides model probabilities. 
    Slow. Using line based verion for notebook.
    """
    token_list = tokenizer.texts_to_sequences([seed_text])
    token_list=token_list[0]
    token_list =pad_sequences([token_list], maxlen= 
                                 max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    predictions=pd.Series(model.predict(token_list, verbose=0)[0]
            ).sort_values(ascending=False)[0:top_n]
    word_dict=tokenizer.word_index
    for i in range(0,len(predictions)):
        word=list(word_dict.keys())[list(
                word_dict.values()).index(predictions.index[i])]
        weight=np.round(predictions.values[i],2)
        print(i, weight, word)
    print('Current sentence: %s'%(seed_text))
    print('What should be the next word?')
    try:
        choice=int(input())
        output_word=list(word_dict.keys())[list(
                word_dict.values()).index(predictions.index[choice])]
        seed_text += " " + output_word
        return seed_text
    except:
        print('ending early')
        return -1
    
    
def augmented_line_writing(model, tokenizer,max_sequence_len, top_n=10, choices=5):
    """
    Allows the user to select next line.
    choices -- number of lines for P.O.E.T to provide as options.
    top_n -- number of optional words for P.O.E.T. to review when selecting the next word in the line (this action is hidden from display)
    """
    #inputText='starttoken starttoken'
    word_dict=tokenizer.word_index
    word_dict2 = {value:key for key, value in word_dict.items()} 
    savedPoem="starttoken"
    b=0
    while b!=-1:
        nextLine=[]
        while len(nextLine)<choices:
            #print('go again')
            inputText=savedPoem
            output_word='';
            while output_word !='returntoken':
                token_list = tokenizer.texts_to_sequences([inputText])[0]
                token_list = pad_sequences([token_list], 
                                       maxlen=max_sequence_len-1, padding='pre')
                predicted = model.predict_classes(token_list, verbose=0)
                #limiting prediction options to just the top n for machine rounding issues.
                predictions=pd.Series(model.predict(token_list, verbose=0)[0]).sort_values(ascending=False)[0:top_n]
                predictions2=(predictions/predictions.sum())
                #rounding for machine precision issues
                predictions2=predictions2.apply(lambda x: np.round(x,2))
                #error is what remains after rounding. Adding to top choice.
                error=1-predictions2.sum()
                predictions2[predictions2.index[0]]=predictions2[predictions2.index[0]]+error
                #random selection of next word using probs from distribution.
                C=np.random.choice(predictions2.index, p=predictions2)
                output_word=word_dict2[C]
                #print(output_word)
                if output_word=='endtoken2':
                    print('End token found. Stopping early.')
                    break
                if output_word=='returntoken':
                    nextLine.append(inputText)
                    nextLine=list(set(nextLine))
                    inputText=savedPoem
                inputText += " " + output_word
        for i in range(0,len(nextLine)):
            print(i, nextLine[i].split('returntoken')[-1])
        print('Current poem:\n%s'%(savedPoem.replace('starttoken','').replace('returntoken','\n ')))
        print('What should be the next line?')
        try:
            choice=int(input())
            line=nextLine[choice]
            savedPoem=savedPoem+' '+line.split('returntoken')[-1]+' returntoken'
            try:
                from IPython.display import clear_output
                clear_output()
            except:
                pass
        except:
            print('ending early')
            b=-1
            try:
                from IPython.display import clear_output
                clear_output()
            except:
                passs
            return savedPoem