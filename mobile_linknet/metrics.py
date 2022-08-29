import tensorflow as tf

@tf.function
def IoU(y_true, y_pred):
    SMOOTH=10.
    true_positive = tf.reduce_sum(y_pred*y_true, axis=[1,2])
    false_positive = tf.reduce_sum(y_pred*(1-y_true), axis=[1,2])
    false_negative = tf.reduce_sum(y_true*(1-y_pred), axis=[1,2])
    return tf.reduce_mean((true_positive+SMOOTH)/(true_positive+false_negative+false_positive+SMOOTH))

@tf.function
def IoU_focal(y_true, y_pred):
    """
    This is similar to IoU loss, but difficult (misclassified) points are more strongly penalized, the extent can be tweaked by gamma. Similar to L1->L2
    The input must have the shape [batch,height,width,channel]
    """
    SMOOTH=10.
    GAMMA=2.
    true_positive = tf.reduce_sum(y_pred*y_true, axis=[1,2])
    false_positive = tf.reduce_sum(tf.math.pow(y_pred*(1-y_true), GAMMA), axis=[1,2])
    false_negative = tf.reduce_sum(tf.math.pow(y_true*(1-y_pred), GAMMA), axis=[1,2])
    return -tf.reduce_mean((true_positive+SMOOTH)/(true_positive+false_negative+false_positive+SMOOTH))

@tf.function
def precision(y_true, y_pred):
    predicted = tf.where(y_pred > 0.5, 1., 0.)
    pred_positive = tf.reduce_sum(predicted, axis=[1,2])
    true_positive = tf.reduce_sum(predicted*y_true, axis=[1,2])

    return tf.reduce_mean(true_positive/(pred_positive+1e-7))

@tf.function
def recall(y_true, y_pred):
    predicted = tf.where(y_pred > 0.5, 1., 0.)
    gt_positive = tf.reduce_sum(y_true, axis=[1,2])
    true_positive = tf.reduce_sum(predicted*y_true, axis=[1,2])
    return tf.reduce_mean(true_positive/(gt_positive+1e-7))

@tf.function
def accuracy(y_true, y_pred):
    predicted = tf.where(y_pred > 0.5, 1., 0.)
    correct_count = tf.reduce_sum(tf.abs(y_true-y_pred), axis=[1,2])
    all_count = y_true.shape[1]*y_true.shape[2]
    return tf.reduce_mean(correct_count/(all_count+1e-7))