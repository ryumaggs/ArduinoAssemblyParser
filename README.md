# Code Base Description and Guide

This is intended for the code bases under **apriori_characterization** and **IntructionTracking** directories.

**All file are written to be used with PyTorch. TensorFlow is not supported**

A description of "parser", which permeates all files, is given under the description of "Main"

## Dependencies
- **Python 2.7**
- PyTorch
- Numpy
- matplotlib
- parser
- Scipy

## Description

You will see the following files:  
- DataLoaders.py
- Graphs.py
- Logger.py
- Main.py
- Networks.py
- Wrapper.py
- misc.py

In addition, you will also see many of their .pyc counter parts, but you only need to edit and run the .py's;  
Do not worry about the .pyc files.

What follows is a description of the important functions in each file. If a function is not mentioned, it either has a low chance of needing change, or I believe to be self explanatory. 

### Main.py
A pretty short file (in terms of length), that will take in all command line arguments, create appropriate dataloader and network, and then pass information to be run in Wrapper.

Here is the command to properly run it via command line. **the ( ) symbols are not needed**:

python Main.py (run name) (GPU_ID) (NetworkClass) --(arg1) (parameters) --(arg2) (parameter1) (parameter2) ... --(argN)

- (run name) is a custom name for the network you are creating
- (GPU_ID) dependent on machine code is housed on, type "nvidia-smi" in command line to see available GPUs
- (Network Class) name of network type you are trying to train
- (arg1) this is an example of a class that takes a single parameter
- (arg) this is an example of a class that takes multiple parameters, all seperated by spaces
- (argN) this is an example of a class that takes no parameters

>Example Runs  
>python Main.py trial 0 AEConv  
>python Main.py trial 0 ClassConv --classifier  
>python Main.py trial 0 ClassConv --classifier --fc_layers 34 19 2 --filter_size 5  

#### Parser

Parser is a python library that allows us to easily manage the command line arguments and "parse" them (for lack of a better term).

Within Main.py, all of the arguments are parsed via the line:

>  args = parser.parse_args(args)

And stored within the variable named "args". This variable will be passed to subsequent files to read run parameters.

If you ever want to see what the parameters of a run are, just print "args". 

As a general policy, if you want to add custom parameters to parser for your own experiments, please tag them with a unique identifier.

For instance, in Main.py, you will see several parameters tagged with "ryu" (my initials) in front. 

### Wrapper.py

This is where the learning and testing code is housed.

Like with all other code, a Wrapper Object is created, and then functions within are called by Object.function_name

I will be presenting the functions not in the order they appear in the file, but in one that willl hopefully facilitate understanding

#### def train(self)

The shell for training a neural network

#### def test(self, load = True)

The shell for loading in a neural network, and testing it

#### def run_epoch(self, data_loader, test=False, ryu_test=False)

This function is called on by both train and test. It calls another function (_iter) that actually forward passes the data through the Neural Network.

This function essentially determines if the network should go through back propogation.

#### def _iter(self, x, y, sumPrint, backwards=True)

This function conducts both forward pass and an optional backwards propogation on the network and the data provided.

### DataLoaders.py

As the name would suggest, this code is responsible for loading in data recordings to the program execution.  
It is written with standard PyTorch DataLoader conventions: The data loader is treated as an object, and thus during  
execution, a DataLoader Object is created and used to read and return data.

The raw data is a long list of floating point values, representing the recorded EM emissions. These raw signals are  
cut into smaller windows (a setable parameter), and these smaller windows are stored in a list, called a "bucket".

> Bucket = [[window1], [window2],...,[windowN]]

N, or the number of files in the bucket, is also a setable parameter

** I will be using the term "Bucket" a lot, so it is best to remember what this is **

####  function __init__(self, args, train), creates a DataLoader Object

Initiallizes a data loader object.  
Depending on the arguments passed in it will only load in files of a particular class. By default it loads in data from the  
class with label 0 (the normal class), but can be passed args such that it loads in data from both normal and anomalous (class 1).  
It reads in the files and divides them into two lists: Training and Test. 

Note: It does not explicitly load in the data, but merely divides the files into groups.

#### function __getitem__(self,index), returns a single item from bucket

If the data bucket is empty, this code will refill the bucket.  
If it is not empty, pop off a single item from the bucket and return

####_handle_file(self,file_name) and _chop_windows(self,data_channels,length)
####returns True (arbitrary) and a mini-bucket of data that will be added to the global data bucket, respectively 

Together, these functions are used to read in a file, chop it into windows, and add the windows to the bucket

### Networks.py

This file contains all of the specifications for different types of network we have used or are going to use.

In class pytorch fashion, each network is a class; the blue prints are in this script.

Each network class will have identical function headers, but different bodies.

#### def __init__(self, args, x_size), creates a NN object

Initialize function for the network.

Layers that are used in the network will be declared and initialized here, but **not** linked.

For instance, if you have an encoding layer and then a fully connected layer, you would declare two variables:

- enc_layer = (something)
- fc_layer = (something)

where (something) specifies the detail of the layer (activation function, connection type, number of nodes)

#### def forward(self, x), returns the output of the final layer of the NN

It is here that the previously established layers are given connections between them. 

Continuing the previous example where we have two layers:
- enc_layer = (something)
- fc_layer = (something)

Here is one way you can connect them:

enc_out = enc_layer(x)  
fc_out = fc_layer(enc_out)  
return fc_out

Notice how I feed the input into one layer, and take the output of that layer as the input to another layer.

The layers can be rearranged in any order

#### def loss(self, x, y, y_bar), returns the loss of the final layer compared to label





