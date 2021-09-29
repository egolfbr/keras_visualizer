from keras.engine.base_layer import Layer
import numpy as np 
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras 
from keras import layers
from keras.layers import Dense, Flatten, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, InputLayer
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
    USAGE = "Usage: python3 script.py -d myModel.dot --db"
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
    myModel.add(Dense(4,activation="relu",input_dim=8))
    myModel.add(Dense(2,activation="tanh"))
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
            myNeurons.append(layer_config["filters"])
        elif layer_type == "Dropout":
            # Look at the input shape (which is the output shape of the previous layer)
            # and apply the dropout rate to it.
            # For example, if the input shape is 6 and the rate is 0.5, that means 50% 
            # of the connections will be turned to 0. We can model this as either 3
            # random connections to random neurons or we can just have 6 connections 
            # and add text to the node describing how that layer acts.
            myNeurons.append(1)
            myDropoutRates.append(layer_config["rate"])
        elif layer_type == "Flatten":
            # Here we would want to represent this as a rectangle, similar to Dropout and it
            # will connect to the following layer with the number of connections equal to 
            # the output shape of the Flatten layer.
            myNeurons.append(1)
        elif layer_type == "Activation":
            # Output number of connections is consistent with the input shape. This layer 
            # does not change the shape of the data, merely transforms it. 
            myNeurons.append(1)
        elif layer_type == "MaxPooling2D":
            myNeurons.append(1)
        elif layer_type == "Input":
            myNeurons.append(1)
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
    dotFile.write("graph[splines=line]\n")
    # Declare all the variable that we will use to create our graph and 
    # and format them.
    for i in range(len(myLayers)):
        if i == 0:
            # This would be the input layer 
            dotFile.write("{\n")
            dotFile.write("node"+ " [shape=circle color=darkgreen]\n")
            dotFile.write("subgraph cluster_0{\n")
            for j in range(myNeurons[i]):
                name = myLayers[i]+"_"+str(j)+"\n"
                dotFile.write(name)
            dotFile.write("label=\"Input Layer\"\n")
            dotFile.write("}\n")
            dotFile.write("}\n")
        elif i == len(myLayers)-1:
            # This would be the output layer
            dotFile.write("{\n")
            dotFile.write("node" + " [shape=circle color=crimson]\n")
            dotFile.write("subgraph cluster_"+str(i)+"{\n")
            for j in range(myNeurons[i]):
                name = myLayers[i]+"_"+str(j)+"\n"
                dotFile.write(name)
            dotFile.write("label=\"Output Layer\"\n")
            dotFile.write("}\n")
            dotFile.write("}\n")
        else:
            # Hidden layers
            dotFile.write("{\n")
            dotFile.write("node" + " [shape=circle color=blue]\n")
            dotFile.write("subgraph cluster_"+str(i)+"{\n")
            for j in range(myNeurons[i]):
                name = myLayers[i]+"_"+str(j)+"\n"
                dotFile.write(name)
            dotFile.write("label=\"Hidden Layer #"+str(i)+"\"\n")
            dotFile.write("}\n")
            dotFile.write("}\n")

    # create a space in the DOT file between declaring all nodes and connecting them
    dotFile.write("\n\n")
    for l in myModel.layers:
        # we obtain the index so that we can easily 
        # access the other lists and arrays for the data that we want
        idx = myModel.layers.index(l)
        # as long as we are not in the last layer we will connect to next layer
        if idx != len(myModel.layers)-1:
        # for each neuron in the current layer, connect it to all neurons in the next layer
            for j in range(myNeurons[idx]):
                for i in range(myNeurons[idx+1]):
                    name = myLayers[idx]+"_"+str(j)
                    name2 = myLayers[idx+1]+"_"+str(i)
                    dotFile.write(name +" -> "+name2+"\n")
        else:
            # we are at output layer
            dotFile.write("}")
            





    dotFile.close()


main()