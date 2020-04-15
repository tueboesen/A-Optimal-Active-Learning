import os
import metrics
import inspect
from main import main

context = {
    'seed': 123557,
    'basename': os.path.basename(__file__).split(".")[0],
    'epochs_AL': 10, #Active learning
    'epochs_SL': 3, #Supervised learning
    'report_pr_epoch': 1,
    'batch_size': 16,
    'lr': 1e-2,
    'use_active_learning_tue': False,
    'use_active_learning': True,
    'use_adaptive_active_learning': True,
    'use_passive_learning': True,
    'use_class_balanced_learning': True,
    'nsamples': 100,
    'nlabels': 0,
    'lr_OED': 1e-5,
    'alpha': 1,
    'sigma': 1,
    'beta': 100,
    'order_dataset': True, #we have this because we have a graph where the classes need to be ordered for the graph to make sense
    'use_label_probabilities': True,   #This will switch the label from a onehot to a probability
    'use_1_vs_all': True,  #Solves the active learning problem in a 1 vs all scenario. Hence each class is pitched against all the others to make sure we probe all the different class boundaries (This should give us points from each decision boundary)
    #Auto encoder
    'use_AE': True,
    'load_AE': '',#'results/autoencoders/10000_linear_10D/autoencoder.pt',
    'lr_AE': 1e-3,
    'decode_dim': 10,
    'epochs_AE': 3,  # Auto encoder
    'network_AE': 'linear'
}

all_metrics = inspect.getmembers(metrics, inspect.isfunction)

for key, value in all_metrics:
    context['metric_name'] = key
    context['metric'] = value
    losses = main(context)

