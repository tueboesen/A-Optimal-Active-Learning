import os
import metrics
import inspect
from main_classifier import main_classifier

context = {
    'seed': 123557,
    'basename': os.path.basename(__file__).split(".")[0],
    'epochs_AL': 10, #Active learning
    'epochs_SL': 100, #Supervised learning
    'report_pr_epoch': 1,
    'batch_size': 16,
    'lr': 1e-2,
    'use_active_learning': True,
    'use_adaptive_active_learning': True,
    'use_passive_learning': True,
    'use_class_balanced_learning': True,
    'nsamples': 10000,
    'nlabels': 0,
    'lr_OED': 1e-5,
    'alpha': 100,
    'sigma': 1,
    'beta': 100,
    'order_dataset': True,  #
    'use_label_probabilities': False,   #This will switch the label from a onehot to a probability
    #Auto encoder
    'use_AE': True,
    'load_AE': 'results/autoencoders/Conv/autoencoder.pt',
    'lr_AE': 1e-3,
    'decode_dim': 32,
    'epochs_AE': 500,  # Auto encoder
}

all_metrics = inspect.getmembers(metrics, inspect.isfunction)

for key, value in all_metrics:
    context['metric_name'] = key
    context['metric'] = value
    losses = main_classifier(context)

