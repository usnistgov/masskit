import numpy as np

def batch_data(x, y, batch_size):
    """
    batch together examples by their length?
    returns
    :param x:
    :param y:
    :param batch_size:
    :return:
    """
    batches = []
    for start in range(0, len(x), batch_size):
        # output from generator must be tuple of (inputs, targets)
        batches.append((np.array(x[start:start+batch_size]), np.array(y[start:start+batch_size])))
    return batches


def batches_from_list(batches):
    """
    generator for keras fit_generator.  Yields (inputs, targets), where inputs and targets are same sized arrays
    batches argument should be a list of x, y pairs,
    where x and y are three-dimensional tensors
    with shape (batch_size, seq_len, 21) and
    (batch_size, seq_len, 3)
    return back numpy arrays
    """
    sz = len(batches)
    i = 0
    while True:
        yield batches[i]
        i += 1
        if i == sz:
            i = 0  # wrap around to beginning
