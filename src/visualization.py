import matplotlib.pyplot as plt
import numpy as np


def select_preview(dataset,dataloader,save=None):
    if dataset == 'mnist':
        pass
    elif dataset == 'circles':
        preview_circles(dataloader,save=save)
    elif dataset == 'cifar10':
        pass
    else:
        raise NotImplementedError("Selected dataset: {}, has not been implemented yet.".format(dataset))
    return

def preview_circles(dataloader,save=None):
    xy = (dataloader.dataset.imgs).numpy()
    labels = dataloader.dataset.labels
    labels_unique = np.unique(labels)
    fig = plt.figure(figsize=[10, 10])
    for label in labels_unique:
        idx = np.where(labels == np.float32(label))[0]
        if label == -1:
            plt.plot(xy[idx, 0], xy[idx, 1], color='gray', marker='o', linestyle="None")
        else:
            plt.plot(xy[idx,0],xy[idx,1],'o')
    plt.title('True classes')
    if save:
        fig.savefig(save)
        plt.close(fig.number)
    return

def plot_results(results,groupid,save=None):
    def plot_err(x, y, label=None):
        y = 100 - np.asarray(y)
        x = np.asarray(x)
        x_mean = np.mean(x, axis=0)
        y_mean = np.mean(y, axis=0)
        plt.semilogy(x_mean, y_mean, '-o', label=label, markersize=3)

    fig_c = plt.figure(figsize=[10, 10])
    fig_ce = plt.figure(figsize=[10, 10])
    plt.clf()
    for result in results:
        if not result['nidx']:
            continue
        x = result['nidx']
        x_mean = np.mean(x, axis=0)

        plt.figure(fig_c.number)
        y=result['cluster_acc']
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        h = plt.plot(x_mean, y_mean, '-o', label=result['method'],gid=groupid)
        plt.fill_between(x_mean, y_mean - y_std, y_mean + y_std, color=h[0].get_color(), alpha=0.2, gid=groupid)
        plt.title('SSL Clustering on train data')
        plt.xlabel('known labels (#)')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.figure(fig_ce.number)
        plot_err(x, y, label=result['method'])

        plt.xlabel('known labels (#)')
        plt.ylabel('Error (%)')
        plt.legend()
        plt.xlim(0, 300)

    if save:
        fileloc = "{}/{}.png".format(save, 'Results_clustering')
        fig_c.savefig(fileloc)
        plt.close(fig_c.number)
        fileloc = "{}/{}.png".format(save, 'Error_clustering')
        fig_ce.savefig(fileloc)
        plt.close(fig_ce.number)
    return


