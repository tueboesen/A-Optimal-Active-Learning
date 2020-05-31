# A-Optimal Active Learning
This code is the official implementation of A-Optimal Active Learning.

To use this code, clone this repository and install the requirements.
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run the following commands:

#### Circles
```train
python train_circles.py 
```
which will produce the following results:
![Alt text](figures/error_circles.png?raw=true "Results")

#### MNIST
```train
python train_mnist.py 
```
which should produce results similar to the following results:
![Alt text](figures/acc_mnist.png?raw=true "Results")

