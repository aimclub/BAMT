
Bayesian block hierarchical learning code is in the folder "block_learning".  

Functionality of the developed code:   

save_bn.py - Function for saving and reading a trained model from file.  

train_bn.py - Function for trainig the structure and parameters of BN from data.  

partial_bn_train.py - Function for adding new BN and trainig new connected BN with hidden vars.  

sampling.py - Function for generating synthetic data from BN model.  

calculate_accuracy.py - Function for estimation of prediction accuracy of BN.

mi_entopy_gauss.py - File with function for MI score calculattion on mixed data.

discretization.py - File with functions of data preprocessing.

quality_metrics.py - Function for quantifying the joint distribution of synthetic data

As a result of the generator operation, a similar hierarchical structure is obtained:
![title](img/BN_gif.gif)




