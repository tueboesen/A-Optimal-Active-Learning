import ast

import matplotlib.pyplot as plt
import numpy as np


def init_results(methods):
    results = []
    for method_name, method_val in methods.items():
        res = setup_result()
        res['method'] = method_name
        results.append(res)
    return results


def setup_result():
    result = {
    'nidx': [],
    'cluster_acc': [],
    'test_acc': [],
    'idx_known': [],
    }
    return result


def update_result(results,idx_known,cluster_acc):
    results['nidx'].append(len(idx_known))
    results['cluster_acc'].append(cluster_acc)
    results['idx_known'].append(idx_known)
    return results

def save_results(results,result,fileloc,j):
    results[j]['nidx'].append(result['nidx'])
    results[j]['cluster_acc'].append(result['cluster_acc'])
    results[j]['test_acc'].append(result['test_acc'])
    results[j]['idx_known'].append(result['idx_known'])
    file_loc = "{}/{}.txt".format(fileloc, results[j]['method'])
    with open(file_loc, 'w') as f:
        for key, value in results[j].items():
            f.write("{:30s} : {} \n".format(key, value))

def load_result(fileloc):
    result = setup_result()
    with open(fileloc, 'r') as f:
        # data = f.read()
        Lines = f.readlines()
        result['nidx'] = ast.literal_eval(Lines[0][33:-2])
        result['cluster_acc'] = ast.literal_eval(Lines[1][33:-2])
        result['test_acc'] = ast.literal_eval(Lines[2][33:-2])
        result['idx_known'] = ast.literal_eval(Lines[3][33:-2])
        method = Lines[4][33:-2]
    return result, method


def plot_result(x,y,title,legend):
    # x = result['nidx']
    x_mean = np.mean(x, axis=0)
    # y=result['cluster_acc']
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    h = plt.plot(x_mean, y_mean, '-o', label=legend, markersize=3)
    plt.fill_between(x_mean, y_mean - y_std, y_mean + y_std, color=h[0].get_color(), alpha=0.2)
    # plt.title(title)
    plt.xlabel('known labels (#)', fontsize=12)
    plt.ylabel('Accuracy (%)')
    plt.legend()
    return



if __name__ == '__main__':
    fileloc_adap = 'E:/Dropbox/ComputationalGenetics/text/Active Learning/Version2/figures/MNIST/active_learning_adaptive.txt'
    fileloc_ms = 'E:/Dropbox/ComputationalGenetics/text/Active Learning/Version2/figures/MNIST/active_learning_ms.txt'
    fileloc_balanced = 'E:/Dropbox/ComputationalGenetics/text/Active Learning/Version2/figures/MNIST/passive_learning_balanced.txt'
    path_out = 'E:/Dropbox/ComputationalGenetics/text/Active Learning/Version2/figures/MNIST/'

    # plt.style.use('E:/Dropbox/ComputationalGenetics/text/Active Learning/Version2/figures/MNIST/style.txt')

    result_adap, method_adap = load_result(fileloc_adap)
    result_ms, method_ms = load_result(fileloc_ms)
    result_balanced, method_balanced = load_result(fileloc_balanced)


    method_adap = 'AL 5'
    method_balanced = 'Balanced'
    method_ms = 'MS 5'
    title = 'Clustering accuracy'
    fig = plt.figure(figsize=[5, 5],dpi=500)
    fig.tight_layout()
    plot_result(result_adap['nidx'],result_adap['cluster_acc'],title,method_adap)
    # plot_result(result_ms,title,method_ms)
    plot_result(result_balanced['nidx'],result_balanced['cluster_acc'],title,method_balanced)
    fileloc = "{}/{}.png".format(path_out, 'Results_clustering')
    fig.savefig(fileloc)
    plt.close(fig.number)



    title = 'Test accuracy'
    fig = plt.figure(figsize=[5, 5], dpi=500)
    plot_result(result_adap['nidx'],result_adap['test_acc'],title,method_adap)
    plot_result(result_balanced['nidx'],result_balanced['test_acc'],title,method_balanced)
    plot_result(result_ms['nidx'],result_ms['test_acc'],title,method_ms)
    fileloc = "{}/{}.png".format(path_out, 'Results_test')
    fig.savefig(fileloc)
    plt.close(fig.number)



    print("done")