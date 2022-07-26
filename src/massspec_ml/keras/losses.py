from tensorflow.keras import backend
import numpy as np
import tensorflow as tf
import sys


# focal loss
# losses from https://github.com/umbertogriffo/focal-loss-keras


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = backend.tf.where(backend.tf.equal(y_true, 1), y_pred, backend.tf.ones_like(y_pred))
        pt_0 = backend.tf.where(backend.tf.equal(y_true, 0), y_pred, backend.tf.zeros_like(y_pred))

        epsilon = backend.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = backend.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = backend.clip(pt_0, epsilon, 1. - epsilon)

        return -backend.sum(alpha * backend.pow(1. - pt_1, gamma) * backend.log(pt_1)) \
               - backend.sum((1 - alpha) * backend.pow(pt_0, gamma) * backend.log(1. - pt_0))

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions_in so that the class probs of each sample sum to 1
        y_pred /= backend.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = backend.epsilon()
        y_pred = backend.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * backend.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * backend.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return backend.sum(loss, axis=1)

    return categorical_focal_loss_fixed


def minibatch_mse():
    """
    minibatched mean square error. returns the square of the average difference between true and predicted.
    This is for when the minibatch consists of conformers of the same molecule
    """

    def minibatch_mse_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        minibatch_mean = backend.mean(y_pred)  # over all axes
        square = backend.square(minibatch_mean - y_true)
        mean = backend.mean(square, axis=-1)
        return mean

    return minibatch_mse_fixed


def triplet_loss(alpha=0.0):
    def triplet_loss_fixed(y_true, y_pred):
        """
        Implementation of the triplet loss function
        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor data
                positive -- the encodings for the positive data (similar to anchor)
                negative -- the encodings for the negative data (different from anchor)
        Returns:
        loss -- real number, value of the loss
        """
        # shape of y_pred should be (,merged layers size)
        #        print_op = tf.print("y_true=", y_true,
        #                            output_stream=sys.stdout)
        #        with tf.control_dependencies([print_op]):
        shape = backend.cast(backend.int_shape(y_pred)[1] / 3, "int32")
        y_pred = backend.print_tensor(y_pred, message="pred=")
        #            y_true = backend.print_tensor(y_true, message="true=")

        anchor = y_pred[:, 0:shape]
        positive = y_pred[:, shape:shape * 2]
        negative = y_pred[:, shape * 2:shape * 3]
        anchor = backend.print_tensor(anchor, message="anchor=")
        negative = backend.print_tensor(negative, message="negative=")

        # distance between the anchor and the positive
        pos_dist = backend.sum(backend.square(anchor - positive), axis=1)
        pos_dist = backend.print_tensor(pos_dist, message="pos_dist=")

        # distance between the anchor and the negative
        neg_dist = backend.sum(backend.square(anchor - negative), axis=1)
        neg_dist = backend.print_tensor(neg_dist, message="neg_dist=")

        # compute loss
        basic_loss = pos_dist - neg_dist + alpha
        basic_loss = backend.print_tensor(basic_loss, message="basic_loss=")
        loss = backend.maximum(basic_loss, 0.0)

        return loss

    return triplet_loss_fixed


def geer_loss():
    def geer_loss_fixed(y_true, y_pred):
        """
        Loss function for a molecular property, between a ground truth anchor, a positive and a negative.
        Requires that the order of the predicted molecular properties follow the same quantitative order
        as the measured molecular properties.  Also requires that the predicted properties match the measured
        properties
        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor data
                positive -- the encodings for the positive data (similar to anchor)
                negative -- the encodings for the negative data (different from anchor)
        Returns:
        loss -- real number, value of the loss
        """
        # shape of y_pred should be (,merged layers size)
        shape = backend.cast(backend.int_shape(y_pred)[1] / 3, "int32")
        y_pred = y_pred * 6362.0
        y_true = y_true * 6362.0

        y_pred = backend.print_tensor(y_pred, message="pred=")
        y_true = backend.print_tensor(y_true, message="true=")

        anchor_pred = y_pred[:, 0:shape]
        positive_pred = y_pred[:, shape:shape * 2]
        negative_pred = y_pred[:, shape * 2:shape * 3]
        #        anchor = backend.print_tensor(anchor, message="anchor=")
        #        negative = backend.print_tensor(negative, message="negative=")

        anchor_true = y_true[:, 0:shape]
        positive_true = y_true[:, shape:shape * 2]
        negative_true = y_true[:, shape * 2:shape * 3]

        mse_anchor = backend.sum(backend.square(anchor_true - anchor_pred), axis=1)
        mse_anchor = backend.print_tensor(mse_anchor, message="mse_anchor=")
        mse_positive = backend.sum(backend.square(positive_true - positive_pred), axis=1)
        mse_positive = backend.print_tensor(mse_positive, message="mse_positive=")
        mse_negative = backend.sum(backend.square(negative_true - negative_pred), axis=1)
        mse_negative = backend.print_tensor(mse_negative, message="mse_negative=")

        # distance between the negative and the positive
        pos_neg_true_dist = positive_true - negative_true
        pos_neg_true_dist = backend.print_tensor(pos_neg_true_dist, message="pos_neg_true_dist=")
        pos_neg_pred_dist = positive_pred - negative_pred
        pos_neg_pred_dist = backend.print_tensor(pos_neg_pred_dist, message="pos_neg_pred_dist=")
        pos_neg_dist = backend.sum(backend.square(pos_neg_pred_dist - pos_neg_true_dist), axis=1)
        pos_neg_dist = backend.print_tensor(pos_neg_dist, message="pos_neg_dist=")

        # distance between the negative and the anchor
        anchor_neg_true_dist = anchor_true - negative_true
        anchor_neg_true_dist = backend.print_tensor(anchor_neg_true_dist, message="anchor_neg_true_dist=")
        anchor_neg_pred_dist = anchor_pred - negative_pred
        anchor_neg_pred_dist = backend.print_tensor(anchor_neg_pred_dist, message="anchor_neg_pred_dist=")
        anchor_neg_dist = backend.sum(backend.square(anchor_neg_pred_dist - anchor_neg_true_dist), axis=1)
        anchor_neg_dist = backend.print_tensor(anchor_neg_dist, message="anchor_neg_dist=")

        # distance between the anchor and the positive
        anchor_pos_true_dist = anchor_true - positive_true
        anchor_pos_true_dist = backend.print_tensor(anchor_pos_true_dist, message="anchor_pos_true_dist=")
        anchor_pos_pred_dist = anchor_pred - positive_pred
        anchor_pos_pred_dist = backend.print_tensor(anchor_pos_pred_dist, message="anchor_pos_pred_dist=")
        anchor_pos_dist = backend.sum(backend.square(anchor_pos_true_dist - anchor_pos_pred_dist), axis=1)
        anchor_pos_dist = backend.print_tensor(anchor_pos_dist, message="anchor_pos_dist=")

        # compute loss
        #        loss = pos_neg_dist + anchor_neg_dist + mse_negative
        loss = mse_positive + mse_anchor + mse_negative + pos_neg_dist + anchor_neg_dist + anchor_pos_dist
        loss = backend.print_tensor(loss, message="loss=")

        return loss

    return geer_loss_fixed


def combo_loss():
    def combo_loss_fixed(y_true, y_pred, alpha=0.0):
        """
        Implementation of the triplet loss function
        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor data
                positive -- the encodings for the positive data (similar to anchor)
                negative -- the encodings for the negative data (different from anchor)
        Returns:
        loss -- real number, value of the loss
        """
        # shape of y_pred should be (,merged layers size)
        print_op = tf.print("y_true=", y_true,
                            output_stream=sys.stdout)
        with tf.control_dependencies([print_op]):
            shape = backend.cast(backend.int_shape(y_pred)[1] / 3, "int32")
            y_pred = backend.print_tensor(y_pred, message="pred=")
            #            y_true = backend.print_tensor(y_true, message="true=")

            anchor = y_pred[:, 0:shape]
            positive = y_pred[:, shape:shape * 2]
            negative = y_pred[:, shape * 2:shape * 3]
            anchor = backend.print_tensor(anchor, message="anchor=")
            negative = backend.print_tensor(negative, message="negative=")

            anchor_true = y_true[:, 0:shape]
            positive_true = y_true[:, shape:shape * 2]
            negative_true = y_true[:, shape * 2:shape * 3]

            mse_negative = backend.sum(backend.square(negative_true - negative), axis=1)
            mse_negative = backend.print_tensor(mse_negative, message="mse_negative=")

            # distance between the anchor and the positive
            pos_dist = backend.sum(backend.square(anchor - positive), axis=1)
            pos_dist = backend.print_tensor(pos_dist, message="pos_dist=")

            # distance between the anchor and the negative
            neg_dist = backend.sum(backend.square(anchor - negative), axis=1)
            neg_dist = backend.print_tensor(neg_dist, message="neg_dist=")

            # compute loss
            basic_loss = pos_dist - neg_dist + alpha
            basic_loss = backend.print_tensor(basic_loss, message="basic_loss=")
            loss = backend.maximum(basic_loss, 0.0) * 100000 + mse_negative
            loss = backend.print_tensor(loss, message="loss=")

        return loss

    return combo_loss_fixed


def e_swish(slope=0.5):
    def e_swish_activation(x):
        return slope * x * backend.sigmoid(x)

    return e_swish_activation

    # def __init__(self, slope=0.5, **kwargs):
    #     self.slope = slope
    #     super(e_swish, self).__init__(**kwargs)
    #
    # def build(self, input_shape):
    #     self.learnable_slope = self.add_weight(shape=(1,),
    #                                            initializer=tf.keras.initializers.Constant(self.slope),
    #                                            trainable=True)
    #     super(e_swish, self).build(input_shape)
    #
    # def call(self, inputs):
    #     return tf.keras.layers.Lambda(lambda x: self.learnable_slope * inputs * backend.sigmoid(inputs))
    #
    # def compute_output_shape(self, input_shape):
    #     return input_shape


# tf.keras.get_custom_objects().update({'e_swish': tf.keras.Activation(e_swish)})


#todo: lambda rank is not finished.  Also, lambda is a gradient, not a loss!
def lambda_rank_loss(sigma=1.0, max_dcg=3.0, direction='DESCENDING'):
    """
    factory for LambdaRank
    :param sigma: is a settable hyperparameter
    :param max_dcg: is the total discounted cumulative gain for a query, but since we are not sampling
        a well distributed chemical space, this is set to a constant for all parameters
    :param direction: the direction of the sort.DESCENDING means that higher document scores are better scores
    :returns: keras loss function
    """

    def lambda_rank(y_true, y_pred):
        """
        LambdaRank, intended to optimize discounted cumulative gain
        using the formulation given in https://research.google/pubs/pub47258/
        assumes a higher document score is a better score
        :param y_true: the true value. If < 0, then do not include in score
        :param y_pred: the predicted value
        :returns: the score
        """
        # for some reason, y_true comes in as a 2d tensor, so flatten
        y_true = backend.flatten(y_true)
        print_op = tf.print("y_true=", y_true, output_stream=sys.stdout)
        with tf.control_dependencies([print_op]):
            y_pred = backend.print_tensor(y_pred, message="y_pred=")
            # first, mask out missing search results
            y_pred = tf.boolean_mask(y_pred, y_true >= 0)
            y_true = tf.boolean_mask(y_true, y_true >= 0)
            # second, sort by y_pred
            y_pred = backend.print_tensor(y_pred, message="y_pred=")
            y_true = backend.print_tensor(y_true, message="y_true=")
            indices = tf.argsort(y_pred, axis=-1, direction=direction)
            indices = backend.print_tensor(indices, message="indices=")
            y_pred = tf.gather(y_pred, indices, axis=-1, name='y_pred_gather')
            y_pred = backend.print_tensor(y_pred, message="y_pred=")
            y_true = tf.gather(y_true, indices, axis=-1, name='y_true_gather')
            y_true = backend.print_tensor(y_true, message="y_true=")
            # create a tensor with indices from 1 to n
            pos = tf.range(0.0, tf.shape(y_pred)[0], 1.0)
            pos = backend.print_tensor(pos, message="pos=")
            # calculate d_i.  The scalar multiply is to scale natural log to log based 2
            d_i = 1.442 * backend.log(pos + 2.0)
            d_i = backend.print_tensor(d_i, message="d_i=")
            g_i = (tf.pow(2.0, y_true) - 1.0) / max_dcg
            g_i = backend.print_tensor(g_i, message="g_i=")

            # create exponent_term matrix by adding an extra axis to y_pred and using broadcastin
            exponent_term = tf.exp(-sigma * (y_pred[:, tf.newaxis] - y_pred[tf.newaxis, :]))
            exponent_term = backend.print_tensor(exponent_term, message="exponent_term=")
            sigmoid_term = 1.442 * backend.log(1 + exponent_term)
            sigmoid_term = backend.print_tensor(sigmoid_term, message="sigmoid_term=")

            # use broadcasting again to create delta_ndcg
            delta_ndcg = tf.abs(g_i[:, tf.newaxis] - g_i[tf.newaxis, :]) * tf.abs(
                1.0 / d_i[:, tf.newaxis] - 1.0 / d_i[tf.newaxis, :])
            delta_ndcg = backend.print_tensor(delta_ndcg, message="delta_ndcg=")
            loss_matrix = delta_ndcg * sigmoid_term
            loss_matrix = backend.print_tensor(loss_matrix, message="loss_matrix=")
            # now do sum over y[i] > y[j]
            # matrix that contains the difference of the true values since we only want to sum if y[i] > y[j]
            y_true_matrix = y_true[:, tf.newaxis] - y_true[tf.newaxis, :]
            y_true_matrix = backend.print_tensor(y_true_matrix, message="y_true_matrix=")
            final_matrix = tf.where(y_true_matrix > 0.0, loss_matrix, tf.zeros(tf.shape(loss_matrix)))
            final_matrix = backend.print_tensor(final_matrix, message="final_matrix=")
            score = backend.sum(final_matrix)
            score = backend.print_tensor(score, message="score=")
        return y_true

    return lambda_rank


def dcg_metric(direction='DESCENDING', tani_and_cos=False):
    """
    factory for discounted cumulative gain
    y_true[i] < 0 indicates no search result at i
    :param direction: the direction of the sort.DESCENDING means that higher document scores are better scores
    :param tani_and_cos: score is tanimoto, but modulated by cosine score
    :returns: keras metric function
    """

    def dcg(y_true, y_pred):
        """
        discounted cumulative gain
        assumes a higher document score is a better score
        :param y_true: the true value. If < 0, then do not include in score
        :param y_pred: the predicted value
        :returns: the score
        """
        if tani_and_cos:
            shape = tf.shape(y_true)[1] // 2
            y_true = y_true[:, 0:shape]

        y_pred = tf.boolean_mask(y_pred, y_true >= 0, name='mask_y_pred')
        y_true = tf.boolean_mask(y_true, y_true >= 0, name='mask_y_true')
        indices = tf.argsort(y_pred, axis=0, direction=direction, name='argsort')
        y_pred = tf.gather(y_pred, indices, axis=0, name='gather_y_pred')
        y_true = tf.gather(y_true, indices, axis=0, name='gather_y_true')
        # pos = tf.range(2.0, backend.int_shape(y_pred)[0]+2.0, 1.0)
        # pos = tf.range(0.0, backend.int_shape(y_pred)[0], 1.0)
        pos = tf.range(0.0, tf.shape(y_pred)[0], 1.0, name='range')
        metric = (backend.pow(2.0, y_true) - 1.0) / (1.442 * backend.log(pos + 2.0))
        return backend.sum(metric)

    return dcg


# the following code modified from tensorflow ranking library, tensorflow_ranking/python/loggers.py and
# tensorflow_ranking/python/losses_impl.py


def approx_ranks(logits, alpha=10.):
    r"""
    Computes approximate ranks given a list of logits.
    Given a list of logits, the rank of an item in the list is simply
    one plus the total number of items with a larger logit. In other words,
      rank_i = 1 + \sum_{j \neq i} I_{s_j > s_i},
    where "I" is the indicator function. The indicator function can be
    approximated by a generalized sigmoid:
      I_{s_j < s_i} \approx 1/(1 + exp(-\alpha * (s_j - s_i))).
    This function approximates the rank of an item using this sigmoid
    approximation to the indicator function. This technique is at the core
    of "A general approximation framework for direct optimization of
    information retrieval measures" by Qin et al.
    Args:
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      alpha: Exponent of the generalized sigmoid function.
    Returns:
      A `Tensor` of ranks with the same shape as logits.
    """
    list_size = tf.shape(input=logits)[1]
    x = tf.tile(tf.expand_dims(logits, 2), [1, 1, list_size])
    y = tf.tile(tf.expand_dims(logits, 1), [1, list_size, 1])
    pairs = tf.sigmoid(alpha * (y - x))
    return tf.reduce_sum(input_tensor=pairs, axis=-1) + .5


def _to_nd_indices(indices):
    """Returns indices used for tf.gather_nd or tf.scatter_nd.
    Args:
      indices: A `Tensor` of shape [batch_size, size] with integer values. The
        values are the indices of another `Tensor`. For example, `indices` is the
        output of tf.argsort or tf.math.top_k.
    Returns:
      A `Tensor` with shape [batch_size, size, 2] that can be used by tf.gather_nd
      or tf.scatter_nd.
    """
    indices.get_shape().assert_has_rank(2)
    batch_ids = tf.ones_like(indices) * tf.expand_dims(
        tf.range(tf.shape(input=indices)[0]), 1)
    return tf.stack([batch_ids, indices], axis=-1)


def sort_by_scores(scores,
                   features_list,
                   topn=None,
                   shuffle_ties=True,
                   seed=None):
    """Sorts example features according to per-example scores.
    Args:
      scores: A `Tensor` of shape [batch_size, list_size] representing the
        per-example scores.
      features_list: A list of `Tensor`s with the same shape as scores to be
        sorted.
      topn: An integer as the cutoff of examples in the sorted list.
      shuffle_ties: A boolean. If True, randomly shuffle before the sorting.
      seed: The ops-level random seed used when `shuffle_ties` is True.
    Returns:
      A list of `Tensor`s as the list of sorted features by `scores`.
    """
    with tf.compat.v1.name_scope(name='sort_by_scores'):
        scores = tf.cast(scores, tf.float32)
        scores.get_shape().assert_has_rank(2)
        list_size = tf.shape(input=scores)[1]
        if topn is None:
            topn = list_size
        topn = tf.minimum(topn, list_size)
        shuffle_ind = None
        if shuffle_ties:
            shuffle_ind = _to_nd_indices(
                tf.argsort(
                    tf.random.uniform(tf.shape(input=scores), seed=seed),
                    stable=True))
            scores = tf.gather_nd(scores, shuffle_ind)
        _, indices = tf.math.top_k(scores, topn, sorted=True)
        nd_indices = _to_nd_indices(indices)
        if shuffle_ind is not None:
            nd_indices = tf.gather_nd(shuffle_ind, nd_indices)
        return [tf.gather_nd(f, nd_indices) for f in features_list]


def inverse_max_dcg(labels,
                    gain_fn=lambda labels: tf.pow(2.0, labels) - 1.,
                    rank_discount_fn=lambda rank: 1. / tf.math.log1p(rank),
                    topn=None):
    """Computes the inverse of max DCG.
    Args:
      labels: A `Tensor` with shape [batch_size, list_size]. Each value is the
        graded relevance of the corresponding item.
      gain_fn: A gain function. By default this is set to: 2^label - 1.
      rank_discount_fn: A discount function. By default this is set to:
        1/log(1+rank).
      topn: An integer as the cutoff of examples in the sorted list.
    Returns:
      A `Tensor` with shape [batch_size, 1].
    """
    ideal_sorted_labels, = sort_by_scores(labels, [labels], topn=topn)
    rank = tf.range(tf.shape(input=ideal_sorted_labels)[1]) + 1
    discounted_gain = gain_fn(ideal_sorted_labels) * rank_discount_fn(
        tf.cast(rank, dtype=tf.float32))
    discounted_gain = tf.reduce_sum(
        input_tensor=discounted_gain, axis=1, keepdims=True)
    return tf.compat.v1.where(
        tf.greater(discounted_gain, 0.), 1. / discounted_gain,
        tf.zeros_like(discounted_gain))


def ndcg(labels, ranks=None, perm_mat=None):
    """Computes NDCG from labels and ranks.
    Args:
      labels: A `Tensor` with shape [batch_size, list_size], representing graded
        relevance.
      ranks: A `Tensor` of the same shape as labels, or [1, list_size], or None.
        If ranks=None, we assume the labels are sorted in their rank.
      perm_mat: A `Tensor` with shape [batch_size, list_size, list_size] or None.
        Permutation matrices with rows correpond to the ranks and columns
        correspond to the indices. An argmax over each row gives the index of the
        element at the corresponding rank.
    Returns:
      A `tensor` of NDCG, ApproxNDCG, or ExpectedNDCG of shape [batch_size, 1].
    """
    if ranks is not None and perm_mat is not None:
        raise ValueError('Cannot use both ranks and perm_mat simultaneously.')

    if ranks is None:
        list_size = tf.shape(labels)[1]
        ranks = tf.range(list_size) + 1
    discounts = 1. / tf.math.log1p(tf.cast(ranks, dtype=tf.float32))
    gains = tf.pow(2., tf.cast(labels, dtype=tf.float32)) - 1.
    if perm_mat is not None:
        gains = tf.reduce_sum(
            input_tensor=perm_mat * tf.expand_dims(gains, 1), axis=-1)
    dcg = tf.reduce_sum(input_tensor=gains * discounts, axis=-1, keepdims=True)
    ndcg_ = dcg * inverse_max_dcg(labels)

    return ndcg_


# The smallest probability that is used to derive smallest logit for invalid or
# padding entries.
_EPSILON = 1e-10


def is_label_valid(labels):
    """Returns a boolean `Tensor` for label validity."""
    labels = tf.convert_to_tensor(value=labels)
    return tf.greater_equal(labels, 0.)


def approx_ndcg_loss(alpha=10.0, include_mse=False):
    """Implements ApproxNDCG loss.
    :param alpha: alpha parameter for sharpness of NDCG approximation
    :param include_mse: include mse?
    """

    def approx_ndcg(labels, logits):
        """See `_RankingLoss`.
            Args:
        labels: A `Tensor` with shape [batch_size, list_size], representing graded
        relevance.  A label below 0 is not valid
        logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
        """
        # todo: deal with batch dim
        labels = tf.transpose(labels, [1, 0])
        labels = backend.print_tensor(labels, 'label')
        logits = tf.transpose(logits, [1, 0])
        logits = backend.print_tensor(logits, 'logits')
        is_valid = is_label_valid(labels)
        labels = tf.compat.v1.where(is_valid, labels, tf.zeros_like(labels))
        if include_mse:
            labels_2 = tf.identity(labels)
            logits_2 = tf.compat.v1.where(is_valid, logits, tf.zeros_like(logits))

        logits = tf.compat.v1.where(
            is_valid, logits, -1e3 * tf.ones_like(logits))  # +
        #    tf.reduce_min(input_tensor=logits, axis=-1, keepdims=True))

        label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
        nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)
        labels = tf.compat.v1.where(nonzero_mask, labels,
                                    _EPSILON * tf.ones_like(labels))
        ranks = approx_ranks(logits, alpha=alpha)

        # # return -ndcg(labels, ranks), tf.reshape(
        # #    tf.cast(nonzero_mask, dtype=tf.float32), [-1, 1])
        # retval_labels = tf.transpose(labels, [1, 0])
        labels = backend.print_tensor(labels, message='processed_labels=')
        # retval_ranks = tf.transpose(ranks, [1, 0])
        ranks = backend.print_tensor(ranks, message='ranks=')

        retval = -ndcg(labels, ranks)
        if include_mse:
            retval = retval + tf.keras.losses.MSE(labels_2, logits_2)
            # also include top 3 ndcg to emphasize the importance of the top few hits
            # short_label = backend.print_tensor(labels[:, 0:3], message="short_label=")
            # retval = retval - ndcg(short_label, ranks[:, 0:3])
        return retval
    return approx_ndcg


def hit_list_loss_factory(include_wasserstein=False, tani_and_cos=False):
    """
    factory to create a loss for a structure search hitlist
    :param include_wasserstein: add an approximation of the wasserstein distance to the loss
    :param tani_and_cos: score is tanimoto, but modulated by cosine score
    """
    def hit_list_loss(labels, logits):
        """
        Loss for a structure search hit list.  The batch is assumed to be the entire hitlist, and ordered
        from best to worst hit.  If label is -1, it means there is no corresponding row in the hitlist
        :param labels: the target tanimoto values.  if tani_and_cos, second part of array is cosine score [0-1.0]
        :param logits: the predicted tanimoto values
        """
        if tani_and_cos:
            shape = tf.shape(labels)[1] // 2
            cos_weight = labels[:, shape:shape * 2]
            labels = labels[:, 0:shape]
        else:
            cos_weight = tf.ones_like(labels)
        cos_weight = tf.squeeze(cos_weight)
        is_valid = is_label_valid(labels)
        labels = tf.compat.v1.where(is_valid, labels, tf.zeros_like(labels))
        labels_2 = tf.identity(labels)
        logits_2 = tf.compat.v1.where(is_valid, logits, tf.zeros_like(logits))
        labels_2 = backend.print_tensor(labels_2, 'labels_2')
        logits_2 = backend.print_tensor(logits_2, 'logits_2')
        mse = tf.keras.losses.MSE(labels_2, logits_2)
        mse = backend.print_tensor(mse, 'mse')
        weights = tf.range(tf.shape(mse)[-1])
        weights = tf.to_float(weights)
        weights = backend.print_tensor(weights, 'weights')
        weights = 1.0/tf.math.log(weights + 2.0)
        weights = backend.print_tensor(weights, 'weights')
        cos_weight = backend.print_tensor(cos_weight, 'cos_weight')
        return_val = mse * weights * cos_weight
        return_val = backend.print_tensor(return_val, 'retval')
        return_val = backend.sum(return_val)
        return_val = backend.print_tensor(return_val, 'sum return_val')

        # ordering
        # logits_2 = tf.transpose(logits_2, [1, 0])
        # diffs = logits_2[0, 0:-1] - logits_2[0, 1:]
        # diffs = backend.print_tensor(diffs, 'diffs')
        # diffs = diffs  # * weights[0:-1]
        # diffs = backend.print_tensor(diffs, 'diffs')
        # diffs = backend.sum(diffs)
        # diffs = backend.print_tensor(diffs, 'diffs')
        # return_val -= diffs
        if include_wasserstein:
            return_val += backend.mean(labels_2 * logits_2)
        return return_val

    return hit_list_loss


def top_hit_metric_factory(direction='DESCENDING', tani_and_cos=False):
    """
    factory to create a metric that returns the highest ranking position of the best or identical hit in a hitlist
    :param direction: direction of the score sort, either DESCENDING or ASCENDING
    :param tani_and_cos: score is tanimoto, but modulated by cosine score

    """
    def top_hit_metric(labels, logits):
        """
        returns position of best hit in search hit list, where one batch is a hit list.
        Assumes at least one hit is available
        :param labels: the ground truth values of dim (num_batch, 1)
        :param logits: the corresponding predicted values of dim (num_batch, 1)
        :return: the position of the best hit, dim (num_batch,)
        """
        if tani_and_cos:
            shape = tf.shape(labels)[1] // 2
            labels = labels[:, 0:shape]

        # get sort indices for the predictions
        indices = tf.argsort(logits, axis=-2, direction=direction)
        # then resort the labels
        gather = tf.squeeze(tf.gather(labels, indices))
        # find the position of the highest scoring label
        ret_val = tf.argmax(gather)

        # # get indices any identical matches (target score > 1)
        # identical_indices = tf.where(tf.greater(tf.squeeze(labels), 1.0))
        # # get the maximal position of the identical hits
        # max_identical = tf.reduce_max(identical_indices)
        # # add in the best hit at position 0
        # max_identical = tf.reduce_max([max_identical, 0])
        # ret_val = tf.reduce_min(indices[0:max_identical+1])

        ret_val = backend.print_tensor(ret_val, 'best hit index')

        return ret_val

    return top_hit_metric
