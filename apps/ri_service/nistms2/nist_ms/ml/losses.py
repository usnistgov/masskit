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
        shape = backend.cast(backend.int_shape(y_pred)[1]/3, "int32")
        y_pred = backend.print_tensor(y_pred, message="pred=")
#            y_true = backend.print_tensor(y_true, message="true=")

        anchor = y_pred[:, 0:shape]
        positive = y_pred[:, shape:shape*2]
        negative = y_pred[:, shape*2:shape*3]
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
