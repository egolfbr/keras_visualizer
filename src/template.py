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
            myNeurons.append(layer_config["units"])
        elif layer_type == "Conv2D":
            myNeurons.append(layer_config["Filters"])
        elif layer_type == "Dropout":
            myDropoutRates.append(layer_config["rate"])
        elif layer_type == "Flatten":
            # Here we would want to represent this as a rectangle, similar to 
            print()
        elif layer_type == "Activation":
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
