Entropy-regularized optimal transport
=====================================

This repository contains the code for the article "Generalized Sinkhorn
iterations for regularizing inverse problems using optimal mass transport" by
J. Karlsson and A. Ringh.

Contents
--------
The code contains the following

* Files containing the implementation of the entropy-regularized optimal
transport and its proximal operator, based on generalized Sinkhorn iterations,
and other utilities.
* Two scripts containing the two examples, and one script for post-processing
some of the images for the hand-example.

Note that the two hand images used in the article do not belong to the authors and are therefore not included.

Installing and running the code
-------------------------------
Clone the repository and install ODL (version 0.6.0) and ASTRA (version 1.8).
This can be done, e.g., by using miniconda run the following commands to set up
a new environment (essentially follow the [odl installation instructions](https://odlgroup.github.io/odl/getting_started/installing.html))
* $ conda create -c odlgroup -n my_env python=3.6 odl=0.6.0 matplotlib pytest scikit-image spyder
* $ source activate my_env
* $ conda install -c astra-toolbox astra-toolbox=1.8

After this, the scripts can be run using, e.g., spyder.

Contact
-------
[Axel Ringh](https://www.kth.se/profile/aringh), PhD student  
Department of Mathematics, KTH Royal Institute of Technology, Stockholm, Sweden  
aringh@kth.se

[Johan Karlsson](http://math.kth.se/~johan79), Associate Professor  
Department of Mathematics, KTH Royal Institute of Technology, Stockholm, Sweden  
johan.karlsson@math.kth.se

