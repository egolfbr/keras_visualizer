from keras import layers
import numpy as np 
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras 
from keras.layers import Dense
from keras.models import Sequential
from keras import Model
import graphviz as gv
import sys 
import subprocess
import optparse
from optparse import OptionParser

def main():
    INFO = "Input a keras model instance and outputs a graphviz DOT file and simulation file"
    VERSION = 0.0
    USAGE = "Usage: python3 script.py -d myModel.dot"
    def showVersion():
        print(INFO)
        print(VERSION)
        print(USAGE)
        sys.exit()

    # parse input arguments
    optparser = OptionParser()
    optparser.add_option("-d","--DOT_FILE",dest="dot_file", default="myModel.dot",help="output DOT file name. Default: myModel.dot")
    optparser.add_option("-v", "--version", action="store_true", dest="showversion",
                         default=False, help="Show the version")
    optparser.add_option("--db", "--debug", action="store_true", dest="debug",
                         default=False, help="Verbose output for debugging")
    (options, filename) = optparser.parse_args()
    if options.showversion:
        showVersion()
    # create a model, in version 1.2 we may want to add this as an input to the script so that the 
    # the script can stand alone 
    myModel = Sequential()
    myModel.add(Dense(4, input_dim=4,activation='relu'))
    myModel.add(Dense(2, activation='relu'))
    myModel.add(Dense(1,  activation='softmax'))


    myLayers = []
    myInputShapes = []
    myOutputShapes = []
    myActivationFunctions = []
    myNeurons = []
    myDropoutRates = []

    for layer in myModel.layers:
        myLayers.append(layer.name)
        myInputShapes.append(layer.input_shape)
        myOutputShapes.append(layer.output_shape)
        tmp = str(layer.output)
        myActivationFunctions.append(tmp.rsplit(" ")[0])
    myDict = myModel.get_config()
    if options.debug == True:
        print("==============Model Dictionary====================")
        print(myDict)
        print("==================================================")
    all_layer_configs = myDict["layers"]
    if options.debug == True:
        print("=================All Layers Dictionary=================")
        print(all_layer_configs)
        print("==================================================")

    for l in all_layer_configs:
        layer_dict = l
        layer_config = layer_dict["config"]
        layer_type = layer_dict["class_name"]
        if options.debug == True:
            print("=================Layer Dictionary=================")
            print(layer_dict)
            print("==================================================")
            print("=================Layer Config Dictionary==========")
            print(layer_config)
            print("==================================================")
        if layer_type == "Dense":
            # Simple 
            myNeurons.append(layer_config["units"])
        elif layer_type == "Conv2D":
            # Convolutional layers do not have neurons, they have filters. 
            # each filter has a specified height and width and is initialized
            # with ther kernel_initializer. The filter is then convoled with the 
            # input image and a new image is formed. Output of this layer is determined
            # based on if padding is "same" or "valid"
            # Ideally I would want to visualized this in the following way:
            # if a layer had 4 filters then I would want 4 neurons of shape square inside 
            # a giant rectangle to show that they are the same layer.
            # Then, the output of that rectangle would be a single connection as a 2D connection 
            # cannot be done in DOT. Or we could implement something similar to what is already 
            # done and just do 1 giant node named "CONV layer" 
            myNeurons.append(layer_config["Filters"])
        elif layer_type == "Dropout":
            # Look at the input shape (which is the output shape of the previous layer)
            # and apply the dropout rate to it.
            # For example, if the input shape is 6 and the rate is 0.5, that means 50% 
            # of the connections will be turned to 0. We can model this as either 3
            # random connections to random neurons or we can just have 6 connections 
            # and add text to the node describing how that layer acts. 
            myDropoutRates.append(layer_config["rate"])
        elif layer_type == "Flatten":
            # Here we would want to represent this as a rectangle, similar to Dropout and it
            # will connect to the following layer with the number of connections equal to 
            # the output shape of the Flatten layer.
            print()
        elif layer_type == "Activation":
            # Output number of connections is consistent with the input shape. This layer 
            # does not change the shape of the data, merely transforms it. 
            print()
        elif layer_type == "MaxPooling2D":
            print()
        else:
            print(layer_type, " is not supported!")

        

    # show model parameters
    if options.debug == True:      
        print("==================Model Parameters==================================")
        print("Layer names:                 ",myLayers)
        print("Input Shape:                 ",myInputShapes)
        print("Output Shape:                ",myOutputShapes)
        print("Activation Function:         ",myActivationFunctions)
        print("Number of neurons per layer: ",myNeurons)
        print("====================================================================")
    



    # Open a DOT file  to write to
    dotFile = open(options.dot_file,'w')
    dotFile.write('digraph g {\n')
    for i in range(len(myLayers)):
        if i == 0:
            # This would be the input layer 
            dotFile.write("{\n")
            dotFile.write("node"+ " [shape=circle color=darkgreen]\n")
            for j in range(myNeurons[i]):
                dotFile.write(myLayers[i]+"_"+str(j)+"\n")
            dotFile.write("}\n")
        elif i == len(myLayers)-1:
            # This would be the output layer
            dotFile.write("{\n")
            dotFile.write("node" + " [shape=circle color=crimson]\n")
            for j in range(myNeurons[i]):
                dotFile.write(myLayers[i]+"_"+str(j)+"\n")
            dotFile.write("}\n")
        else:
            # Hidden layers
            dotFile.write("{\n")
            dotFile.write("node" + " [shape=circle color=blue]\n")
            for j in range(myNeurons[i]):
                dotFile.write(myLayers[i]+"_"+str(j)+"\n")
            dotFile.write("}\n")
    
    

    
        

    dotFile.write("}")
    dotFile.close()




main()
