import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import sys 


# takes a keras model instance as input
def writedotfile(myModel,debug=False,fileName = "myDotFile.dot"):
    """ 
    Makes DOT file from keras model instance. 

    Parameters
    ----------
    myModel : Keras model instance
    debug : Boolean, optional
        A flag used to determine if code will have verbose output
        Default: False
    filename : str, optional
        String to the path and filename of where you want to save the DOt file to 
        Default: ./myDotFile.dot 
    
    Returns
    -------
    Nothing
    """
    INFO = "Input a keras model instance and outputs a graphviz DOT file."
    VERSION = 0.2
    def showVersion():
        print(INFO)
        print(VERSION)
        sys.exit()
    if debug:
        showVersion()    
    # create a model, in version 1.2 we may want to add this as an input to the script so that the 
    # the script can stand alone 
    rnnLayers = ["lstm","gru","simpleRNN"]

    myLayers = []
    myInputShapes = []
    myOutputShapes = []
    myActivationFunctions = []
    myNeurons = []
    myDropoutRates = []
    mySplits = []
    for layer in myModel.layers:
        layer_name = layer.name
        splits = layer_name.split('_')
        mySplits.append(splits[0])
        myLayers.append(layer.name)
        myInputShapes.append(layer.input_shape)
        myOutputShapes.append(layer.output_shape)
        tmp = str(layer.output)
        myActivationFunctions.append(tmp.rsplit(" ")[0])
    myDict = myModel.get_config()
    if debug == True:
        print("==============Model Dictionary====================")
        print(myDict)
        print("==================================================")
    all_layer_configs = myDict["layers"]
    if debug == True:
        print("=================All Layers Dictionary=================")
        print(all_layer_configs)
        print("==================================================")

    for l in all_layer_configs:
        layer_dict = l
        layer_config = layer_dict["config"]
        layer_type = layer_dict["class_name"]
        if debug == True:
            print("=================Layer Dictionary=================")
            print(layer_dict)
            print("==================================================")
            print("=================Layer Config Dictionary==========")
            print(layer_config)
            print("==================================================")
        if layer_type == "Dense":
            myNeurons.append(layer_config["units"])
        elif layer_type == "Conv2D":
            myNeurons.append(layer_config["filters"])
        elif layer_type == "Dropout":
            myNeurons.append(1)
            myDropoutRates.append(layer_config["rate"])
        elif layer_type == "Flatten":
            myNeurons.append(1)
        elif layer_type == "Activation":
            myNeurons.append(1)
        elif layer_type == "MaxPooling2D":           
            myNeurons.append(1)
        elif layer_type == "MaxPooling3D":
            myNeurons.append(1)
        elif layer_type == "MaxPooling1D":
            myNeurons.append(1)
        elif layer_type == "AveragePooling2D":
            myNeurons.append(1)
        elif layer_type == "AveragePooling3D":
            myNeurons.append(1)
        elif layer_type == "AveragePooling1D":
            myNeurons.append(1)
        elif layer_type == "GlobalMaxPooling2D":
            myNeurons.append(1)
        elif layer_type == "GlobalMaxPooling3D":
            myNeurons.append(1)
        elif layer_type == "GlobalMaxPooling1D":
            myNeurons.append(1)
        elif layer_type == "GlobalAveragePooling2D":
            myNeurons.append(1)
        elif layer_type == "GlobalAveragePooling3D":
            myNeurons.append(1)
        elif layer_type == "GlobalAveragePooling1D":
            myNeurons.append(1)
        elif layer_type == "Input":
            myNeurons.append(1)
        elif layer_type == "Conv3D":
            myNeurons.append(layer_config["filters"])
        elif layer_type == "LSTM":
            myNeurons.append(layer_config["units"])
        elif layer_type == "SimpleRNN":
            myNeurons.append(layer_config["units"])
        elif layer_type == "GRU":
            myNeurons.append(layer_config["units"])
        else:
            print(layer_type, " is not supported!")
    # show model parameters
    if debug == True:      
        print("==================Model Parameters==================================")
        print("Layer names:                 ",myLayers)
        print("Input Shape:                 ",myInputShapes)
        print("Output Shape:                ",myOutputShapes)
        print("Activation Function:         ",myActivationFunctions)
        print("Number of neurons per layer: ",myNeurons)
        print("====================================================================")
    # Open a DOT file  to write to
    dotFile = open(fileName,'w')
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
            if(mySplits[idx] in rnnLayers):
                for j in range(myNeurons[idx]):
                        nm = myLayers[idx]+"_"+str(j)
                        dotFile.write(nm +" -> "+nm+"[dir=back]\n")
                        for i in range(myNeurons[idx+1]):
                            name = myLayers[idx]+"_"+str(j)
                            name2 = myLayers[idx+1]+"_"+str(i)
                            dotFile.write(name +" -> "+name2+"\n") 
            else:
                for j in range(myNeurons[idx]):
                    for i in range(myNeurons[idx+1]):
                        name = myLayers[idx]+"_"+str(j)
                        name2 = myLayers[idx+1]+"_"+str(i)
                        dotFile.write(name +" -> "+name2+"\n")
        else:
            # we are at output layer
            dotFile.write("}")
    dotFile.close()


def neuron_viewer(myTrainedModel,layer_num=0,neuron_num=0,input_num=0,fileName="myNeuron.dot",debug=False):

    myLayers = []
    myInputShapes = []
    for layer in myTrainedModel.layers:
        myLayers.append(layer.name)
        myInputShapes.append(layer.input_shape)
    myDict = myTrainedModel.get_config()
    layer = myTrainedModel.layers[layer_num]
    layer_weights = layer.get_weights()[0]
    rows = layer_weights.shape[0]-1
    layer_biases = layer.get_weights()[1]
    neuronDotFile = open(fileName,'w')
    neuronDotFile.write("digraph g {\n")
    neuronDotFile.write("graph[splines=line]\n\"")
    neuronDotFile.write(myLayers[layer_num])
    neuronDotFile.write("\\n")
    neuronDotFile.write("weight = " + str(layer_weights[input_num][neuron_num]))
    neuronDotFile.write("\n")
    neuronDotFile.write("bias = " + str(layer_biases[neuron_num]) + "\"")
    neuronDotFile.write("}")
    neuronDotFile.close()

def neuron_group_viewer(myTrainedModel,layer_num=0,begin_neuron=0,end_neuron=2,input_num=0,fileName="myNeurons.dot",debug=False):

    myLayers = []
    myInputShapes = []
    for layer in myTrainedModel.layers:
        myLayers.append(layer.name)
        myInputShapes.append(layer.input_shape)
    layer = myTrainedModel.layers[layer_num]
    layer_weights = layer.get_weights()[0]
    rows = layer_weights.shape[0]-1
    layer_biases = layer.get_weights()[1]
    neuronDotFile = open(fileName,'w')
    neuronDotFile.write("digraph g {\n")
    neuronDotFile.write("graph[splines=line]\n")
    if (end_neuron - begin_neuron <= rows):
    	for neuron in range(begin_neuron,end_neuron+1):
    		neuronDotFile.write("\"" + myLayers[layer_num] + "_" + str(neuron))
    		neuronDotFile.write("\\n")
    		neuronDotFile.write("weight = " + str(layer_weights[input_num][neuron]))
    		neuronDotFile.write("\\n")
    		neuronDotFile.write("bias = " + str(layer_biases[neuron]) + "\"\n")
    	neuronDotFile.write("}")
    	neuronDotFile.close()
    return 0

if __name__ == "writedotfile":
    writedotfile()
    
if __name__ == "neuron_group_viewer":
    neuron_group_viewer()

if __name__ == "neuron_viewer":
    neuron_viewer()
