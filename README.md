## A-Optimal Active Learning

This repository contains the source code accompanying the paper:

[A-Optimal Active Learning](https://arxiv.org/abs/2110.09585) 

*We present an approach that is based on A-optimal experimental design of ill-posed problems and show how one can optimally label a
data set by partially probing it, and use it to train a deep network. We present two approaches that
make different assumptions on the data set. The first is based on a Bayesian interpretation of the
semi-supervised learning problem with the graph Laplacian that is used for the prior distribution and
the second is based on a frequentist approach, that updates the estimation of the bias term based on
the recovery of the labels. We demonstrate that this approach can be highly efficient for estimating
labels and training a deep network.*

## Dependencies

This code is based on PyTorch and requires the following packages:

* torch
* numpy
* scipy
* hnswlib

The exact environment that the paper was run with exist in `requirement.txt`, but the repository has since been updated further to make it easier for others to use and adapt by improving the documentation and adding type hints to critical functions.

## Experiments
In the paper we run experiements on circle disks and MNIST. In this repository we have provided simple runner scripts that will emulate those experiments (the results won't be reproduced exactly since the code has been altered since the results in the paper were produced, but comparable results should be produced). 

### Circle-disks
This is a simple 2D example that highlights the active learning problem that we are trying to showcase and solve. It is highly recommended to run this problem to get an understanding of how the code works.

`run_circles.py` will create a dataset with 3 circle-disks with different labels, with only one label known from each disk. The runtime of this example is relatively quick and should be done in a few minutes on most modern computers.

![Alt text](figures/True_classes.png?raw=true "True classes")

![Alt text](figures/Results_clustering.png?raw=true "Results")


### MNIST
This is a classic benchmark dataset that has previously been used in active learning. The MNIST example has some additional steps that separates it from the circle-disk example that might make it attractive to run in order to get a deeper understanding. 
1) The MNIST example incorporates feature extraction using an autoencoder. This was not needed in the circle-disk example, since the coordinate distance between each point was already the perfect metric for the distance function required by the graph Laplacian, but for images like MNIST there is no such perfect measure.
2) The MNIST example utilizes a neural network for predicting pseudo-labels on top of the clustering, which was not used in the simpler circle-disk example.
3) The MNIST example compares its method to the psudo-labelling approach suggested by [K. Wang et Al.](https://arxiv.org/abs/1701.03551). 

`run_MNIST.py` will download and run MNIST, using all 60000 training samples and 10000 testing samples. The runtime of this example is very long and will likely take between a week and a month to run on a normal computer. However, the code can easily be sped up by lowering the amount of training/testing samples, and the number of training epochs the neural network runs after each active learning step.


## Cite
If you found this work useful in your research, please consider citing:
```
@article{boesen2021optimal,
  title={A-Optimal Active Learning},
  author={Boesen, Tue and Haber, Eldad},
  journal={arXiv preprint arXiv:2110.09585},
  year={2021}
}
```

## License

The code and scripts in this repository are distributed under MIT license. See LICENSE file.
