import numpy as np

def compact(series):
    '''
    Compact Time Series.
    '''
    compacted = []
    pre = series[0]
    compacted.append(pre)
    for e in series[1:]:
        if e != pre:
            pre = e
            compacted.append(e)
    return compacted

def remove_duplication(series):
    '''
    Remove duplication.
    '''
    result = []
    for e in series:
        if e not in result:
            result.append(e)
    return result
    
def seg_to_label(label):
    pre = 0
    seg = []
    for l in label:
        seg.append(np.ones(l-pre,dtype=int)*label[l])
        pre = l
    result = np.concatenate(seg)
    return result

def reorder_label(label):
    # Start from 0.
    label = np.array(label)
    ordered_label_set = remove_duplication(compact(label))
    idx_list = [np.argwhere(label==e) for e in ordered_label_set]
    for i, idx in enumerate(idx_list):
        label[idx] = i
    return label

def adjust_label(label):
    '''
    Adjust label order.
    '''
    label = np.array(label)
    compacted_label = compact(label)
    ordered_label_set = remove_duplication(compacted_label)
    label_set = set(label)
    idx_list = [np.argwhere(label==e) for e in ordered_label_set]
    for idx, elem in zip(idx_list,label_set):
        label[idx] = elem
    return label

def bucket_vote(bucket):
    '''
    The bucket vote algorithm.
    @return: element of the largest amount.
    @Param bucket: the bucket of data, array like, one dim.
    '''
    vote_vector = np.zeros(len(set(bucket)), dtype=int)
    
    # create symbol table
    symbol_table = {}
    symbol_list = []
    for i, s in enumerate(set(bucket)):
        symbol_table[s] = i
        symbol_list.append(s)

    # do vote
    for e in bucket:
        vote_vector[symbol_table[e]] += 1

    symbol_idx = np.argmax(vote_vector)
    return symbol_list[symbol_idx]

def smooth(X, bucket_size):
    for i in range(0,len(X), bucket_size):
        s = bucket_vote(X[i:i+bucket_size])
        true_size = len(X[i:i+bucket_size])
        X[i:i+bucket_size] = s*np.ones(true_size,dtype=int)
    return X

def dilate_label(label, f, max_len):
    slice_list = []
    for e in label:
        slice_list.append(e*np.ones(f, dtype=int))
    return np.concatenate(slice_list)[:max_len]

def str_list_to_label(label):
    label_set = remove_duplication(label)
    label = np.array(label)
    new_label = np.array(np.ones(len(label)))
    for i, l in enumerate(label_set):
        idx = np.argwhere(label==l)
        new_label[idx] = i
    return new_label.astype(int)