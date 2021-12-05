# keras_visualizer
A front-end keras visualizer

## Description
This is a front-end, keras model visualizer. Existing visualizers only work within Tensorflow or other back-end wrappers, however, in an effort
to eliminate or reduce vendor-lock scenarios, we though that it would be a good idea to create a front-end visualizer. There is already a front-end
solution available [here](https://github.com/lordmahyar/keras-visualizer) but there is limited functionality. We aim to expand on this work by adding
more layers and output formats but most importantly, adding simulation type functionality to the visualizer to allow designers to see how their data 
is transformed though the neural network. 

# Getting Started
## Install Keras
We used plaidml as our backend. This allows us to use whatever computational tool our computer has regardless if it is NVIDIA or AMD. 
```
> pip install plaidml-keras
```
After installation, setup up plaidml to work with your computers hardware. Follow the prompts to select which device you want to run your models on. 
```
> plaidml-setup
```

Next you will have to make sure that keras is set to use the proper backend. If you already have Keras you can use one of the following options to change the backend environment from tensorflow to plaidml. If you installed Keras for the first time using the above command, I believe these steps are unneccessary but you may want to double check just in case by using option 2. 

Option 1: Inline change

In your python script at the very top before any code is written put the following two lines 
```python
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
```
This will change the variable ```KERAS_BACKEND``` to use plaidML in the ```keras.json``` file

Option 2: Manually edit the JSON file 

This option is also straightforward. First locate the ```keras.json ``` file in the ```.keras``` folder which is usually in your home directory. Once located, you can open and change the backend variable to use plaidML and then you do not need to include the lines of code in option 1. 
```
{
"floatx" : "float32",
"epsilon" : 1e-07,
"backend" : "plaidml.keras.backend",
"image_data_format" : "channels_last"
}
```
An important side note, when working with image data, make sure your data matches the "image_data_format" tag or you can change the tag to match your data. If they do not match, you will get very poor results from your model. 

Once this change has been made refresh your workspace and open a terminal and run the setup command again to ensure that plaidML is working properly.

Now you are ready to install the keras_dot_visualizer!

# Installation and Usage
The work for this is maintained in GitHub and as a result there are two ways you can get the code. You can manually download from the repository, or you can install via pip. 

Example install via pip:
```
python3 -m pip install git+https://github.com/egolfbr/keras_visualizer.git
```

Example usage: 
```
from keras_dot_visualizer import writedot
from writedot import writedotfile
...
...
writedotfile(your_keras_model_instance)
```

# Future Work
In order to complete this project in time, we had to sacrifice some functionality that would be very useful to have. Hopefully in future versions we can add functionality for custom activation functions, custom layers, sparsely connected layers and many more. We are also currently working on a neuron viewer which will be able to simulate a single neuron or a group of neurons on a given layer, given some input data. 
### Contact information
Brian Egolf - egolfbr@miamioh.edu 

Jonathan Hagan - haganjd2@miamioh.edu
