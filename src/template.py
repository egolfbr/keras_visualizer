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
    (options, filename) = optparser.parse_args()
    if options.showversion:
        showVersion()
    # create a model, in version 1.2 we may want to add this as an input to the script so that the 
    # the script can stand alone 
    myModel = Sequential()
    myModel.add(Dense(16, input_dim=16,activation='relu'))
    myModel.add(Dense(12, activation='relu'))
    myModel.add(Dense(4,  activation='softmax'))


    myLayers = []
    myInputShapes = []
    myOutputShapes = []
    myActivationFunctions = []
    myNeurons = []

    for layer in myModel.layers:
        myLayers.append(layer.name)
        myInputShapes.append(layer.input_shape)
        myOutputShapes.append(layer.output_shape)
        tmp = str(layer.output)
        myActivationFunctions.append(tmp.rsplit(" ")[0])
    myDict = myModel.get_config()
    all_layer_configs = myDict["layers"]
    for l in all_layer_configs:
        layer_dict = l
        layer_config = layer_dict["config"]
        #print(layer_config["units"])
        myNeurons.append(layer_config["units"])

    # show model parameters      
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
            dotFile.write(myLayers[i] + " [shape=circle color=darkgreen]\n")
            for j in range(myNeurons[i]):
                dotFile.write(myLayers[i]+"_"+str(j)+"\n")
            dotFile.write("}\n")
        elif i == len(myLayers)-1:
            # This would be the output layer
            dotFile.write("{\n")
            dotFile.write(myLayers[i] + " [shape=circle color=crimson]\n")
            for j in range(myNeurons[i]):
                dotFile.write(myLayers[i]+"_"+str(j)+"\n")
            dotFile.write("}\n")
        else:
            # Hidden layers
            dotFile.write("{\n")
            dotFile.write(myLayers[i] + " [shape=circle color=white]\n")
            for j in range(myNeurons[i]):
                dotFile.write(myLayers[i]+"_"+str(j)+"\n")
            dotFile.write("}\n")
    
    

    
        

    dotFile.write("}")
    dotFile.close()
main()
