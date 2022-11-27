import torch
import torch.nn.functional as F


def run_active_learning_ms(dl,idx_labels,c,net,nlabels,delta):
    """
    An active learning approach suggested in:
    Keze Wang et al. “Cost-effective active learning for deep image classification”. In: IEEE Transactions on Circuits and Systems for Video Technology 396 27.12 (2016), pp. 2591–2600.

    The method ranks unlabeled samples in an ascending order according to the first and second most probable class labels as predicted by the classifier,
    and selects the most uncertain ones for labelling. The psuedo-labelling is done by a process called high-confidence psuedo-labelling,
    which assigns a label to those datapoints for which the classifiers labeling entropy is less than a threshold value δ.
    :param dl: dataloader
    :param idx_labels: indices of already known labels
    :param c: configuration parameters
    :param net: neural network
    :param nlabels: number of datapoints to label
    :param delta: threshold parameter
    :return:
    """
    # First we predict labels for all of dl_org and find the ones that are most uncertain
    device = c.device
    net.eval()
    ms_ref = torch.ones(nlabels,device=device)
    idx_ms = -torch.ones(nlabels,dtype=torch.int64,device=device)
    idx_pseudo = []
    label_pseudo = []

    with torch.no_grad():
        for images, labels, _, idx in dl:
            images = images.to(device)
            idx = idx.to(device)
            # labels = labels.to(device)
            outputs = net(images)
            prob = F.softmax(outputs, dim=1)

            # This finds the samples that we need to label
            val, label = torch.topk(prob,2) #find the ms
            ms = val[:,0]- val[:,1]
            ms_tot = torch.cat([ms,ms_ref])
            idx_tot = torch.cat([idx,idx_ms])
            _, indices = torch.topk(ms_tot,nlabels,largest=False)
            idx_ms = idx_tot[indices]
            ms_ref = ms_tot[indices]

            # This finds the high confidence samples
            # We need to calculate the cross entropy
            ce = - torch.sum(prob * torch.log(prob),dim=1)
            M = ce < delta
            # idx in idx_labels
            idx_pseudo.append(idx[M])
            label_pseudo.append(label[M,0])


    idx_pseudo = torch.cat(idx_pseudo).tolist()
    label_pseudo = torch.cat(label_pseudo).tolist()
    idx_labels = idx_labels + idx_ms.tolist()

    #now remove all idx_pseudo that already exist in idx_labels
    idx_pseudo_dict = dict((k, i) for i, k in enumerate(idx_pseudo))
    inter_diff = set(idx_pseudo).difference(idx_labels)
    indices = [idx_pseudo_dict[x] for x in inter_diff]

    idx_pseudo = [idx_pseudo[indi] for indi in indices]
    label_pseudo = [label_pseudo[indi] for indi in indices]



    return idx_labels,idx_pseudo,label_pseudo
