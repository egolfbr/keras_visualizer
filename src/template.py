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

    # for layer in myModel.layers:
    #     print("Layer Name: "+layer.name)
    #     myModel.get_layer(layer.name)
    #     print("Layer input shape: ")
    #     print(layer.input_shape)
    #     print("Layer Output shape: ")
    #     print(layer.output_shape)
    #     print("Layer activation: ")
    #     print(layer.output)
    #     tmp = str(layer.output)
    #     print(tmp.rsplit(" ")[0])
    #     print("-------------------")

    myLayers = []
    myInputShapes = []
    myOutputShapes = []
    myActivationFunctions = []
    for layer in myModel.layers:
        myLayers.append(layer.name)
        myInputShapes.append(layer.input_shape)
        myOutputShapes.append(layer.output_shape)
        tmp = str(layer.output)
        myActivationFunctions.append(tmp.rsplit(" ")[0])
    print(myLayers)
    print(myInputShapes)
    print(myOutputShapes)
    print(myActivationFunctions)
    myDict = myModel.get_config()
    #print(myDict.values())
    for x in myDict.values():
        print(x,"\n")
main()
