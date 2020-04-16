import os

from src.main import main

context = {
    'seed': 123557,                                         #Seed for reproduceability
    'basename': os.path.basename(__file__).split(".")[0],   #Names the folder after this filename
    #Determines which type of learning to test:
    'use_active_learning_tue': False,                       #An active learning approach that tries to optimize the information gained by each point it selects
    'use_active_learning': True,                            #Bayesian active learning
    'use_adaptive_active_learning': True,                   #Adaptive active learning
    'use_passive_learning': True,                           #Passive learning with randomly selected points
    'use_class_balanced_learning': True,                    #Passive learning with class balanced selection of points
    #Dataset
    'nsamples': 10000,                                      #Number of samples in dataset
    'order_dataset': True,                                  #Orders the samples in dataset by class (they still get shuffled when used by a dataloader)
    'nlabels': 0,                                           #Number of labels to start with in an adaptive scheme
    'use_label_probabilities': True,                        #Switch the labels from onehot to a probabilities
    'batch_size': 16,
    #Supervised learning
    'epochs_SL': 10,                                         #Epochs for supervised learning
    'lr': 1e-2,

    #Active learning
    'epochs_AL': 10,                                        #Iterations to use in Active learning
    'lr_AL': 1e-5,
    'nlabels_pr_epoch_pr_class': 5,                         #Number of labels to learn in each iteration per class
    'alpha': 1,
    'sigma': 1,
    'beta': 100,
    'use_1_vs_all': True,

    #Auto encoder
    'use_AE': True,                                        #Use an autoencoder to generate an encoded feature space
    'load_AE': '',#'results/autoencoders/10000_linear_10D/autoencoder.pt',
    'epochs_AE': 500,
    'lr_AE': 1e-3,
    'network_AE': 'linear',                                 #Options are 'conv','linear'
    'decode_dim': 10,                                       #When network is linear this determines the dimension of the encoded space.
}

losses = main(context)

