# keras_visualizer
A front-end keras visualizer

## Description
This is a front-end, keras model visualizer. Existing visualizers only work within Tensorflow or other back-end wrappers, however, in an effort
to eliminate or reduce vendor-lock scenarios, we though that it would be a good idea to create a front-end visualizer. There is already a front-end
solution available [here](https://github.com/lordmahyar/keras-visualizer) but there is limited functionality. We aim to expand on this work by adding
more layers and output formats but most importantly, adding simulation type functionality to the visualizer to allow designers to see how their data 
is transformed though the neural network. 

## Getting Started
You may want to start a virtual environment that includes the packages that you want to work with. This can be done using the virtualenv tool that is available 
to download using ```pip```. After creatint an environment you can install many of the common pip packages that machine learning engineers use such as ```pandas```,```numpy```, ```matplotlib```,```seaborn```, and ```scikit-learn```. These can be installed by using the ```pip``` or ```pip3``` command followed by the package name.

Example:

```pip install pandas```

After installing those dpendencies you will need to install keras. Keras is included in the tensorflow package but since we are focusing on front-end development we do not want tensorflow. Thus, we are left with two options. First we can install keras all alone. This can be done as follows ```pip install keras```. The other option is to install keras with a backend that allows for different GPUs. This is ideal because it will allow us to build and simulate. This can be done by using the following command ```pip install plaidml-keras```. PlaidML is an open-source keras backend that allows for users to use different GPUs other than Nvidia (which is required by tensorflow). After you install PlaidML you will need to edit the ```keras.json``` file. You can do this two different ways. 

Option 1: Inline change

In your python script at the very top before any code is written put the following two lines 
```python
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
```
This will change the variable ```KERAS_BACKEND``` to use plaidML in the ```keras.json``` file

Option 2: Manually edit the JSON file 

This option is also straightforward. First locate the ```keras.json ``` file in the ```.keras``` folder which is usually in your home directory. Once located, you can open and change the backend variable to use plaidML and then you do not need to include the lines of code in option 1. 

Once this change has been made refresh your workspace and open a terminal and run the following command: 
```plaidml-setup```

This will run and require user input in setting up some the parameters for the plaidML backend. This is not important in this project since we are focusing on visualization of machine learning models and not optimizing them. The setup script basically asks the user which device to execute the code on (CPU, GPU, Integrated graphics ect).

Now you are ready to install the keras_dot_visualizer!

## Installation and Usage
The work for this is maintained in GitHub and as a result there are two ways you can get the code. You can manually download from the repository, or you can install via pip.

```
python -m pip install https://github.com/ObeyedSky622/keras_visualizer
```
## Future Work
In order to complete this project in time, we had to sacrifice some functionality that would be very useful to have. Hopefully in future versions we can add functionality for custom activation functions, custom layers, sparsely connected layers and many more.
### Contact information
Brian Egolf - egolfbr@miamioh.edu 

Jonathan Hagan - haganjd2@miamioh.edu
