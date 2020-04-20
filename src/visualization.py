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




def preview(X, Ytarget, Y):
    shift, scale = findimageshift(X, Ytarget, Y)
    plt.figure(figsize=[10, 8]);
    plt.subplot(1, 3, 1);
    plt.title('Input')
    imshow2(X, shift=shift, scale=scale)

    plt.subplot(1, 3, 2);
    plt.title('Target')
    imshow2(Ytarget,shift=shift, scale=scale)

    plt.subplot(1, 3, 3);
    plt.title('Output')
    imshow2(Y,shift=shift, scale=scale)

    plt.pause(0.4)

def plot_results(fh,results,legend,save=None):
    plt.figure(fh.number)
    plt.subplot(2,1,1)
    plt.plot(results['nidxs'],results['cluster_acc'],'-o',label=legend)
    plt.title('SSL Clustering  (%)')
    plt.xlabel('known labels (#)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(results['nidxs'],results['learning_acc'],'-o',label=legend)
    plt.title('Learning Accuracy (%)')
    plt.xlabel('known labels (#)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    if save:
        fileloc = "{}/{}.png".format(save, 'Results')
        fh.savefig(fileloc)



