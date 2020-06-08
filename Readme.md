In this project, the pyBN library was used to work with Bayesian networks (author Nicholas Cullen <ncullen.th@dartmouth.edu>)


Link to the repository with the library:https://github.com/ncullen93/pyBN

Our Bayesian block hierarchical learning code is in the folder "block_learning".  

Functionality of the developed code:  
save_bn.py - Function for structural and parametric training of a separate Bayesian network with subsequent saving to a json. file  

read_bn.py - Function for reading a trained model from a json. file  

connect_BNs.py - Function for connecting a trained Bayesian network model to a new network (trained on new data) through a hidden variable.  

sampling.py - Function for sampling data from a Bayesian network at any level of the block structure.





