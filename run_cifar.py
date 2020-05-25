import argparse
import os

from src.main import main

parser = argparse.ArgumentParser(description='A-Optimal Active Learning')

#General
parser.add_argument('--seed', default=123557, type=int, metavar='N',help='seed number')
parser.add_argument('--basefolder', default=os.path.basename(__file__).split(".")[0], type=str, metavar='N',help='Basefolder where results are saved')
parser.add_argument('--nrepeats', default=1, type=int, metavar='N',help='Number of times the code will be rerun')
parser.add_argument('--mode', default='fast', type=str, metavar='N',help='Mode to run in (debug,fast,paper)')
#Data
parser.add_argument('--dataset', default='cifar10', type=str, metavar='N',help='Name of dataset to run, currently implemented: (circles,mnist,cifar10)')
parser.add_argument('--nsamples', default=1000, type=int, metavar='N',help='Number of datasamples')
parser.add_argument('--nlabels', default=20, type=int, metavar='N',help='Number of labels to start with')
parser.add_argument('--initial-labeling-mode', default='balanced', type=str, metavar='N',help='Modes for selecting initial labeled points (balanced,random,bayesian)')
parser.add_argument('--batch-size', default=16, type=int, metavar='N',help='batch size used in dataloader')
#Feature Transform
parser.add_argument('--feature-transform', default='', type=str, metavar='N',help='Type of feature transform to use on data before computing graph Laplacian')
#Graph Laplacian
parser.add_argument('--L-metric', default='l2', type=str, metavar='N',help='Type of metric the graph Laplacian is computed with (l2,cosine)')
parser.add_argument('--L-knn', default=10, type=int, metavar='N',help='Number of nearest neighbours to include in L')
parser.add_argument('--L-tau', default=1e-2, type=float, metavar='N',help='Hyperparameter in the computation of the regularization =(L + tau*I)^eta')
parser.add_argument('--L-eta', default=2, type=int, metavar='N',help='Hyperparameter in the computation of the regularization =(L + tau*I)^eta')
#Active Learning
parser.add_argument('--AL-types', default=0, type=float, metavar='N',help='Hyperparameter (not used in the current implementation)')
parser.add_argument('--AL-iterations', default=3, type=int, metavar='N',help='Number of active learning iterations to run')
parser.add_argument('--AL-nlabels-pr-class', default=3, type=int, metavar='N',help='Number of data points to label for each class iteration')
parser.add_argument('--AL-alpha', default=1, type=float, metavar='N',help='Hyperparameter')
parser.add_argument('--AL-beta', default=0, type=float, metavar='N',help='Hyperparameter (not used in the current implementation)')
parser.add_argument('--AL-sigma', default=1e-2, type=float, metavar='N',help='Hyperparameter')
parser.add_argument('--AL-w0', default=1e6, type=float, metavar='N',help='Hyperparameter, sets the weight of each labeled datapoint to this value')
#Learning
parser.add_argument('--SL-epochs', default=5, type=int, metavar='N',help='Number of epochs to train the network')
parser.add_argument('--SL-network', default='resnet', type=str, metavar='N',help='select the neural network to train (resnet)')
parser.add_argument('--SL-loss-type', default='MSE', type=str, metavar='N',help='Loss type for network, (MSE or CE)')



AL_methods = {
    'active_learning_bayesian': False,  # Bayesian active learning
    'active_learning_adaptive': True,  # Adaptive active learning
    'passive_learning': True,  # Passive learning with randomly selected points
    'passive_learning_balanced': True,  # Passive learning with class balanced selection of points
              }
args = parser.parse_args()
# state = {k: v for k, v in args._get_kwargs()}
args.AL_methods = AL_methods
losses = main(args)

