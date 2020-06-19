# Reproduction of Relational Fusion Network
This library contains a reproducted implementation of the Relational Fusion Network (RFN) 
introduced in the SIGSPATIAL'19 paper _Graph Convolutional Networks for Road Networks_ 
by Tobias Skovgaard Jepsen, Christian S. Jensen, and Thomas Dyhre Nielsen. 
Paper can be found in [here](https://arxiv.org/abs/1908.11567) 
and code can be found in [here](https://github.com/TobiasSkovgaardJepsen/relational-fusion-networks) .
Most of the code is from the original code.
Custom added codes are in custom folder.
Data is in `data.zip`. First unzip it.
Example network parameter data is in `model_data` folder.
See `Train.ipynb` for training, `Test.ipynb` for testing.

You need to install osmnx, MXNet, dgl, PyTorch.
MXNet and dgl should be gpu-version.