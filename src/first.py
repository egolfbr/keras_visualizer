# Pretty useless, just here to demoonstrate how to use the import statements and 
# how we can write to output files as well as list characteristics of each layer
import numpy as np 
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras 
from keras.layers import Dense
from keras.models import Sequential
from keras import Model
model_name = "my_first_model"

fileOut = open(model_name+".json",'w')

myModel = Sequential()
myModel.add(Dense(16, input_dim=16,activation='relu'))
myModel.add(Dense(12, activation='relu'))
myModel.add(Dense(4,  activation='softmax'))

old = ","
new = ",\n"
fileOut.write(Model.to_json(myModel).replace(old,new))
#print(len(myModel.layers))
for layer in myModel.layers:
    print("Layer Name: "+layer.name)
    myModel.get_layer(layer.name)
    print("Layer input shape: ")
    print(layer.input_shape)
    print("Layer Output shape: ")
    print(layer.output_shape)
    print("-------------------")
#print(type(myModel.summary()))
