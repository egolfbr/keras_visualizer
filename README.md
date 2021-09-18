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

After installing those dpendencies you will need to install keras. Keras is included in the tensorflow package but since we are focusing on front-end development we do not want tensorflow. Thus, we are left with two options. First we can install keras all alone. This can be done as follows ```pip install keras```. The other option is to install keras with a backend that allows for different GPUs. This is ideal because it will allow us to build and simulate. This can be done by using the following command ```pip install plaidml-keras```. PlaidML is an open-source keras backend that allows for users to use different GPUs other than Nvidia (which is required by tensorflow).


### Contact information
Brian Egolf - egolfbr@miamioh.edu 

Jonathan Hagan - haganjd2@miamioh.edu
