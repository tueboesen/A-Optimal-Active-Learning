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

