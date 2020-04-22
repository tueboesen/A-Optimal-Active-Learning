import os

from src.main import main

context = {
    'seed': 123557,                                         #Seed for reproduceability
    'basename': os.path.basename(__file__).split(".")[0],   #Names the folder after this filename
    'nrepeats': 3,                                          #Sets how many times the code should be repeated
    #Determines which type of learning to test:
    'AL_methods': {
        'active_learning_bayesian': False,  # Bayesian active learning
        'active_learning_adaptive': True,  # Adaptive active learning
        'passive_learning': True,  # Passive learning with randomly selected points
        'passive_learning_balanced': True,  # Passive learning with class balanced selection of points
    },
    #Dataset
    'nsamples': 1000,                                        #Number of samples in dataset
    'order_dataset': True,                                  #Orders the samples in dataset by class (they still get shuffled when used by a dataloader)
    'nlabels': 20,                                           #Number of labels to start with in an adaptive scheme
    'use_label_probabilities': True,                        #Switch the labels from onehot to a probabilities
    'batch_size': 16,
    'use_1_vs_all_dataset': 0,                              #If negative the 1_vs_all_dataset is not used, otherwise the selected number will be pitched against all other labels
    #Supervised learning
    'use_SL': False,
    'epochs_SL': 3,                                         #Epochs for supervised learning
    'lr': 1e-2,
    'loss_type': 'MSE',                                     #Options are MSE and CE
    # Grap Laplacian
    'metric': 'l2',  # 'l2' or 'cosine'
    'knn': 50,  # Number of nearest neighbours

    #Active learning
    'epochs_AL': 3,                                        #Iterations to use in Active learning
    'lr_AL': 1e-3,
    'nlabels_pr_class': 2,                                  #Number of labels to learn in each iteration
    'alpha': 0.1,
    'sigma': 1,
    'beta': 100,
    'use_1_vs_all': True,
    'recompute_L': True,                                    #Switch features to the output from the network and recompute it each iteration.

    #Auto encoder
    'use_AE': True,                                        #Use an autoencoder to generate an encoded feature space
    'load_AE': '',#'results/autoencoders/10000_linear_10D/autoencoder.pt',
    'epochs_AE': 0,
    'lr_AE': 1e-3,
    'network_AE': 'linear',                                 #Options are 'conv','linear'
    'decode_dim': 10,                                       #When network is linear this determines the dimension of the encoded space.
}

losses = main(context)

