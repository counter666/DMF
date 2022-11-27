def dc(result, reference):
    result = np.atleast_1d(result.astype(np.bool))  
    reference = np.atleast_1d(reference.astype(np.bool))  

    tp = np.count_nonzero(result & reference)
    fp = np.count_nonzero(result & ~reference)
    tn = np.count_nonzero(~result & ~reference)
    fn = np.count_nonzero(~result & reference)

    try:
        dc = 2.0 * tp / float(fn+fp+2*tp)
    except ZeroDivisionError:
        dc = 0.0

    return dc
  
def jc(result, reference):
   
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    
    tp = np.count_nonzero(result & reference)
    fp = np.count_nonzero(result & ~reference)
    tn = np.count_nonzero(~result & ~reference)
    fn = np.count_nonzero(~result & reference)

    jc = tp / float(fp+fn+tp)

    return jc
  
def recall(result, reference):
    
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def sensitivity(result, reference):
   
    return recall(result, reference)


def specificity(result, reference):
    
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)

    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0

    return specificity

  
def accuracy(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    fp = np.count_nonzero(result & ~reference)
    tn = np.count_nonzero(~result & ~reference)
    fn = np.count_nonzero(~result & reference)

    try:
        accuracy = (tp + tn) / float(tp + fn + tn + fp)
    except ZeroDivisionError:
        accuracy = 0.0

    return accuracy
