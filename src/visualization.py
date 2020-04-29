import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch

def imshow(img):
    #This assumes that images are within range -1:1
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def imshow2(imgs,shift=0,scale=1):
    '''
    We visualize images using a cyclic representation of colors, we do this such that values which are close will end up with similar colors, but we don't actually care about which color they end with
    :param img:
    :return:
    '''
    imgs = (imgs+shift)/scale
    plt.imshow(torchvision.utils.make_grid(imgs,4).permute(1, 2, 0).detach().numpy()) #imshow expect images to be within 0:1 in range for floats or 0:255 in integer

def imshow3(imgs,shift=0,scale=1):
    '''
    We visualize images using a cyclic representation of colors, we do this such that values which are close will end up with similar colors, but we don't actually care about which color they end with
    :param img:
    :return:
    '''
    imgs = (imgs+shift)/scale
    plt.imshow(torchvision.utils.make_grid(imgs,4).permute(1, 2, 0).detach().numpy()) #imshow expect images to be within 0:1 in range for floats or 0:255 in integer


#
def findimageshift(*imgs):
    '''
    This function finds a global shift among a group of images, such that all image values are between 0 and 1
    :param imgs:
    :return:
    '''
    images = torch.cat(imgs, dim=0)
    shift = - np.min(torch.min(images).item(),0)
    scale = torch.max(images+shift).item()
    return shift,scale

# def findimagescale(*imgs):
#     '''
#     This function finds the scale needed to transform a set of images to have values between -1 and 1
#     :param imgs: input images
#     :return: scale (scalar)
#     '''
#     images = torch.cat(imgs, dim=0)
#     scale = torch.max(torch.abs(images)).item()
#     return scale




def preview(dataloader,save=None):
    if dataloader.dataset.name == 'circles':
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


def vizualize_circles(U,dataset,saveprefix=None,iter=None):
    xy = (dataset.imgs).numpy()
    labels = dataset.labels
    plabels = U
    plabels = plabels - np.min(plabels,axis=1)[:,None]
    psum = np.sum(plabels,axis=1)
    idx = np.where(psum != np.float32((0)))
    plabels[idx] = plabels[idx]/psum[idx,None]
    nc = dataset.nc
    fig = plt.figure(figsize=[10, 10])
    plt.scatter(xy[:,0], xy[:,1], c=plabels, alpha=0.5)
    idx = np.where(labels != np.float32(-1))[0]
    plt.scatter(xy[idx,0], xy[idx,1], c=plabels[idx,:],marker='x', s=100)
    if saveprefix:
        save = "{}_{}_{}.png".format(saveprefix, iter,'cluster')
        fig.savefig(save)
        plt.close(fig.number)
    return



def debug_circles(xy,df,idx_known,y,idx_new,saveprefix=None):
    def plot_and_color(xy,y,title):
        n = y.shape[0]
        if y.ndim == 2:
            if y.shape[1] == 3:
                y = y - np.min(y, axis=1)[:, None]
                ysum = np.sum(y, axis=1)
                idx = np.where(ysum != np.float32((0)))
                y[idx] = y[idx] / ysum[idx, None]
                plt.scatter(xy[:, 0], xy[:, 1], c=y)
        else:
            ym = np.max(np.abs(y))
            yn = y / ym
            rgba_colors = np.zeros((n, 4))
            rgba_colors[yn > 0, 0] = 1
            rgba_colors[:, 3] = np.abs(yn) #THIS IS WHY IT IS ALPHA
            plt.scatter(xy[:,0], xy[:,1], c=rgba_colors)
            # idx=np.argsort(np.abs(y),)[::-1]
            # plt.scatter(xy[idx[0:5],0], xy[idx[0:5],1], marker='d', s=100)
            # rgba_colors = np.zeros((n, 3))
            # idx = y > 0
            # rgba_colors[idx, 0] = y[idx]/np.max(y)
            # idx = y < 0
            # rgba_colors[idx, 1] = y[idx]/np.min(y)

    fig = plt.figure(3,figsize=[20, 10])
    plt.clf()
    plt.subplot(1,2,1)
    plot_and_color(xy,df,'df visualized')
    plt.scatter(xy[list(idx_new),0],xy[list(idx_new),1],marker='+',s=150)
    plt.subplot(1,2,2)
    y = np.squeeze(y)
    plot_and_color(xy,y,'labels visualized')
    # plt.pause(1)
    if saveprefix:
        save = "{}_{}.png".format(saveprefix, 'AL')
        fig.savefig(save)
        plt.close(fig.number)
    return


def plot_results(results,groupid,save=None):
    fig_c = plt.figure(figsize=[10, 10])
    fig_l = plt.figure(figsize=[10, 10])
    fig_v = plt.figure(figsize=[10, 10])
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

        plt.figure(fig_l.number)
        y=result['learning_acc']
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        h = plt.plot(x_mean, y_mean, '-o', label=result['method'],gid=groupid)
        plt.fill_between(x_mean, y_mean - y_std, y_mean + y_std, color=h[0].get_color(), alpha=0.2, gid=groupid)
        plt.title('Network on train set')
        plt.xlabel('known labels (#)')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.figure(fig_v.number)
        y=result['validator_acc']
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        h = plt.plot(x_mean, y_mean, '-o', label=result['method'],gid=groupid)
        plt.fill_between(x_mean, y_mean - y_std, y_mean + y_std, color=h[0].get_color(), alpha=0.2, gid=groupid)
        plt.title('Network on validator set')
        plt.xlabel('known labels (#)')
        plt.ylabel('Accuracy (%)')
        plt.legend()


    if save:
        fileloc = "{}/{}.png".format(save, 'Results_clustering')
        fig_c.savefig(fileloc)
        plt.close(fig_c.number)
        fileloc = "{}/{}.png".format(save, 'Results_network_train')
        fig_l.savefig(fileloc)
        plt.close(fig_l.number)
        fileloc = "{}/{}.png".format(save, 'Results_network_validate')
        fig_v.savefig(fileloc)
        plt.close(fig_v.number)
    return


