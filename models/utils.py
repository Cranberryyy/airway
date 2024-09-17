import tensorflow as tf
import os
from tensorflow.keras.metrics import binary_accuracy
import numpy as np
from tensorflow.keras.layers import Conv3D, MaxPooling3D
import nibabel as nib
import keras.backend as K
# from .tools.sampling import farthest_point_sample, gather_point
# from .tools.emover_distance import approx_match, match_cost
# from .tools.chamfer_distance import nn_distance
epsilon = 1e-07


"""
Create connectivity 
Convert connectivity to binary prediction
"""
#### create connectivity
kernel_ = np.zeros([3, 3, 3, 1, 27])  #H W D
c = 0
for i in range(3): #H
    for j in range(3): #D
        for k in range(3): #W
            kernel_[i,j,k,0,c] = 1
            kernel_[1,1,1,0,c] = 1
            c += 1
kernel_ = tf.keras.initializers.Constant(kernel_)

def create_connectivity(input, name=None):
    if name is None:
        name = 'create_connectivity'
    output = Conv3D(27, 3, use_bias=False, padding = 'SAME', kernel_initializer=kernel_, trainable=False, name=name)(input)
    output = output / 2
    return output

#### inverse connectivity
kernel = np.zeros([3, 3, 3, 27, 27])
c = 26
for i in range(3):
    for j in range(3):
        for k in range(3):
            kernel[i, j, k, c, c]  = 1
            kernel[1,1,1, c,c] =1
            c -= 1
kernel = tf.keras.initializers.Constant(kernel)

def inverse_connectivity(input_, name=None):
    if name is None:
        name = 'inverse_connectivity'
    output = Conv3D(27, 3, use_bias=False, padding = 'SAME', kernel_initializer=kernel, trainable=False, name=name)(input_)
    output = output / 2
    return output



def create_distance(label, name=None):
    label_dilate = soft_dilate(label)
    boundary = label_dilate - label
    delta = label * 0
    for i in range(1, 10, 1):
        boundary_dilate = soft_dilate(boundary)
        delta_1 = label * (boundary_dilate - boundary) 
        delta_2 = delta_1 * i
        boundary = boundary_dilate
        delta = delta + tf.math.minimum(delta_2 + delta, delta_2)
    outline_to_centerline = delta
    return outline_to_centerline

def create_boundary(label, name=None):
    label_dilate = soft_dilate(label)
    boundary = label_dilate - label
    delta = label * 0
    for i in range(1, 2, 1):
        boundary_dilate = soft_dilate(boundary)
        delta_1 = label * (boundary_dilate - boundary) 
        delta_2 = delta_1 * i
        boundary = boundary_dilate
        delta = delta + tf.math.minimum(delta_2 + delta, delta_2)
    return delta

def soft_erode(img):
    p1 = -MaxPooling3D(pool_size=(3, 3, 1), strides=(1, 1, 1), padding='same', data_format=None)(-img)
    p2 = -MaxPooling3D(pool_size=(3, 1, 3), strides=(1, 1, 1), padding='same', data_format=None)(-img)
    p3 = -MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 1, 1), padding='same', data_format=None)(-img)
    return tf.math.minimum(tf.math.minimum(p1, p2), p3)

def soft_dilate(img):
    return MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same', data_format=None)(img)

def soft_open(img):
    img = soft_erode(img)
    img = soft_dilate(img)
    return img

def soft_skel(img, iters):
    img1 = soft_open(img)
    skel = tf.nn.relu(img-img1)
    for j in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = tf.nn.relu(img-img1)
        intersect = tf.math.multiply(skel, delta)
        skel += tf.nn.relu(delta-intersect)
    return skel

def extract_outline_centerline(input):
    centerline = soft_skel(input, 15)
    outline = input - soft_erode(input)
    return outline, centerline





"""
Loss functions
Loss function for point clouds
"""
def earth_mover(pcd1, pcd2, num_points):
    assert pcd1.shape[1] == pcd2.shape[1]
    match = approx_match(pcd1, pcd2)
    cost = match_cost(pcd1, pcd2, match)
    return tf.reduce_mean(tf.math.divide_no_nan(cost, num_points))

def chamfer(pcd1, pcd2):
    dist1, _, dist2, _ = nn_distance(pcd1, pcd2)
    # import pdb;pdb.set_trace()
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    #return (dist1 + dist2) / 2.0
    return dist1 + dist2


def chamfer_breakage(pcd1, pcd2):
    dist1, _, dist2, _ = nn_distance(pcd1, pcd2)
    # import pdb;pdb.set_trace()
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = dist1  #  tf.reduce_mean(tf.sqrt(dist2))
    #return (dist1 + dist2) / 2.0
    return dist1 + dist2

def calc_cd(output, gt, separate=False, return_raw=False):
    dist1, idx1, dist2, idx2= nn_distance(gt, output)
    cd_p = (tf.sqrt(tf.reduce_mean(dist1, axis=1)) + tf.sqrt(tf.reduce_mean(dist2,axis=1))) / 2
    cd_t = (tf.reduce_mean(dist1, axis=1) + tf.reduce_mean(dist2, axis=1))

    if separate:
        res = [tf.concat([tf.sqrt(dist1).mean(axis=1, keepdims=True), tf.sqrt(dist2).mean(axis=1, keepdims=True)], axis=0),
               tf.concat([dist1.mean(axis=1, keepdims=True), dist2.mean(axis=1, keepdims=True)], axis=0)]
    else:
        res = [cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res

def density_chamfer(pcd1, pcd2, T=1000, n_p=1, return_raw=False, separate=False, return_freq=False, non_reg=False):
    gt = tf.cast(pcd1, tf.float32)
    x = tf.cast(pcd2, tf.float32)
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True, separate=separate)
    exp_dist1, exp_dist2 = tf.exp(-dist1 * T), tf.exp(-dist2 * T)

    loss1 = []
    loss2 = []
    gt_counted = []
    x_counted = []

    for b in range(batch_size):
        count1 = tf.math.bincount(idx1[b])
        weight1 = tf.cast(tf.gather(count1, idx1[b], axis=0), tf.float32) ** n_p
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1.append(tf.reduce_mean(- exp_dist1[b] * weight1 + 1.))

        count2 = tf.math.bincount(idx2[b])
        weight2 = tf.cast(tf.gather(count2, idx2[b], axis=0), tf.float32) ** n_p
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2.append(tf.reduce_mean(- exp_dist2[b] * weight2 + 1.))

        if return_freq:
            expand_count1 = tf.zeros_like(idx2[b])  # n_x
            expand_count1 = tf.tensor_scatter_nd_update(expand_count1, tf.expand_dims(tf.range(count1.shape[0]), axis=1), count1)
            x_counted.append(expand_count1)
            expand_count2 = tf.zeros_like(idx1[b])  # n_gt
            expand_count2 = tf.tensor_scatter_nd_update(expand_count2, tf.expand_dims(tf.range(count2.shape[0]), axis=1), count2)
            gt_counted.append(expand_count2)

    loss1 = tf.stack(loss1)
    loss2 = tf.stack(loss2)
    loss = (loss1 + loss2) / 2
    return tf.reduce_mean(loss)


def distance_loss(gt_dist, pred_dist):
    gt_outline_to_centerline, gt_centerline_to_outline = gt_dist
    pred_outline_to_centerline, pred_centerline_to_outline = pred_dist
    abs_dist_outline_to_centerline = tf.abs(tf.exp(-gt_outline_to_centerline) - tf.exp(-pred_outline_to_centerline))
    # abs_dist_centerline_to_outline = tf.abs(tf.exp(-gt_centerline_to_outline) - tf.exp(-pred_centerline_to_outline))
    #dist_loss = (tf.reduce_mean(abs_dist_centerline_to_outline) + tf.reduce_mean(abs_dist_outline_to_centerline)) / (tf.reduce_mean(tf.cast(gt_outline_to_centerline > 0, tf.float32)) + 1)
    dist_loss = (tf.reduce_mean(gt_outline_to_centerline)) / (tf.reduce_mean(tf.cast(gt_outline_to_centerline > 0, tf.float32)) + 1)
    return dist_loss

def boundary_loss(gt_boundary,pred_boundary):
    gt_boundary = tf.cast(gt_boundary >0, tf.float32)
    pred_boundary = tf.cast(pred_boundary >0, tf.float32)
    boundary_loss = dice_loss_func(gt_boundary, pred_boundary)
    return boundary_loss


def dice_loss_func(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true*y_pred)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true + y_pred) + smooth)
    dice_coef = tf.reduce_mean(dice)
    return 1 - dice_coef

def soft_clDice_loss(y_true, y_pred, iter_ = 15):
    smooth = 1.
    skel_pred = soft_skel(y_pred, iter_)
    skel_true = soft_skel(y_true, iter_)
    pres = (K.sum(tf.math.multiply(skel_pred, y_true))+smooth)/(K.sum(skel_pred)+smooth)    
    rec = (K.sum(tf.math.multiply(skel_true, y_pred))+smooth)/(K.sum(skel_true)+smooth)    
    cl_dice_loss = 1.- 2.0*(pres*rec)/(pres+rec)
    return cl_dice_loss


def balanced_binary_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            name='binary_crossentropy')(y_true, y_pred)
    return bce


def cal_weights(y_true, y_pred, gama=1):
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_true_argmax = tf.cast(tf.expand_dims(tf.argmax(y_true, axis=-1),axis=-1), tf.float32)
    B = tf.keras.layers.AveragePooling3D(pool_size=3, strides=1, padding='same')(y_true_argmax)
    C = tf.math.sign(y_true_argmax) * B
    D = y_true_argmax - C
    E = soft_dilate(D)
    weights = (tf.keras.layers.AveragePooling3D(pool_size=3, strides=1, padding='same')(E))*gama + 1
    return weights

def cal_recall_weights(y_true, y_pred):
    pred = tf.reshape(tf.argmax(y_pred, axis=-1), [-1])
    gt = tf.reshape(tf.argmax(y_true, axis=-1), [-1])
    gt_idx, _, gt_count= tf.unique_with_counts(gt)
    idex = (pred != gt)
    fn = pred[idex]
    fn_idx, _, fn_count = tf.unique_with_counts(fn)
    # idex = (pred == gt)
    # tp = pred[idex]
    #tp_idx, _, tp_count = tf.unique_with_counts(tp)
    weight = tf.cast(fn_count, tf.float32) / (tf.cast(gt_count, tf.float32)+1)
    return weight

def weighted_binary_crossentropy(y_true, y_pred, gama=1, is_recall=False):
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    voxel_weights = cal_weights(y_true, y_pred, gama)
    loss = (-y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)) * voxel_weights
    if is_recall:
        recall_weights = cal_recall_weights(y_true, y_pred)
        loss = loss * tf.nn.softmax(recall_weights)
    return tf.reduce_mean(loss)


def balanced_categorical_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    cce = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            name='category_crossentropy')(y_true, y_pred)
    return cce

def abs_loss(y_true, y_pred):
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)

def kl_loss(y_true, y_pred):
    return tf.keras.losses.KLDivergence()(y_true, y_pred)

def weighted_abs_loss(y_true, y_pred, gama=1):
    weights = cal_weights(y_true, y_pred, gama)
    return tf.reduce_mean(tf.math.abs(y_true - y_pred) * weights)


def focal_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    return tf.keras.losses.BinaryFocalCrossentropy(
                gamma=2.0, from_logits=False)(y_true, y_pred)




"""
Evaluation metrics
"""

def be_binary_accuracy(y_true, y_pred):
    class_nums = y_pred.shape[-1]//2
    y_true = y_true[..., class_nums:]
    y_pred = y_pred[..., class_nums:]
    bi_acc = binary_accuracy(y_true, y_pred)
    return bi_acc



def jaccard_index(y_true, y_pred, smooth=1, threshold=0.5):
    prediction = tf.where(y_pred > threshold, 1, 0)
    prediction = tf.cast(prediction, dtype=y_true.dtype)
    ground_truth_area = tf.reduce_sum(
        y_true, axis=(1, 2, 3))
    prediction_area = tf.reduce_sum(
        prediction, axis=(1, 2, 3))
    intersection_area = tf.reduce_sum(
        y_true*y_pred, axis=(1, 2, 3))
    union_area = (ground_truth_area
                    + prediction_area
                    - intersection_area)
    jaccard = tf.reduce_mean(
        (intersection_area + smooth)/(union_area + smooth))
    return jaccard

def numeric_score(pred, gt):
    FP = float(np.sum((pred == 1) & (gt == 0)))
    FN = float(np.sum((pred == 0) & (gt == 1)))
    TP = float(np.sum((pred == 1) & (gt == 1)))
    TN = float(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN

def numeric_score(pred, gt):
    FP = float(np.sum((pred == 1) & (gt == 0)))
    FN = float(np.sum((pred == 0) & (gt == 1)))
    TP = float(np.sum((pred == 1) & (gt == 1)))
    TN = float(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN

def metrics_3d(pred, gt):
    FP, FN, TP, TN = numeric_score(pred, gt)
    tpr = TP / (TP + FN + 1e-10)
    fnr = FN / (FN + TP + 1e-10)
    fpr = FN / (FP + TN + 1e-10)
    iou = TP / (TP + FN + FP + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    return tpr, fnr, fpr, iou, precision


def tree_length_calculation(pred, label_skeleton, smooth=1e-5):
    pred = pred.flatten()
    label_skeleton = label_skeleton.flatten()
    tree_length_ratio = (np.sum(pred * label_skeleton) + smooth) / (np.sum(label_skeleton) + smooth)
    return tree_length_ratio

def branch_detected_calculation(pred, label_parsing, label_skeleton, thresh=0.8):
    pred = np.int16(pred); label_parsing = np.int16(label_parsing); label_skeleton = np.int16(label_skeleton)
    label_branch = label_skeleton * label_parsing
    label_branch_flat = label_branch.flatten()
    label_branch_bincount = np.bincount(label_branch_flat)[1:]
    total_branch_num = label_branch_bincount.shape[0]
    pred_branch = label_branch * pred
    pred_branch_flat = pred_branch.flatten()
    pred_branch_bincount = np.bincount(pred_branch_flat)[1:]
    if total_branch_num != pred_branch_bincount.shape[0]:
        lack_num = total_branch_num - pred_branch_bincount.shape[0]
        pred_branch_bincount = np.concatenate((pred_branch_bincount, np.zeros(lack_num)))
    branch_ratio_array = pred_branch_bincount / label_branch_bincount
    branch_ratio_array = np.where(branch_ratio_array >= thresh, 1, 0)
    detected_branch_num = np.count_nonzero(branch_ratio_array)
    detected_branch_ratio = round((detected_branch_num * 100) / total_branch_num, 2)
    return total_branch_num, detected_branch_num, detected_branch_ratio

def dice_coef(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    intersection = np.sum(pred * label)
    dice_coefficient_score = round(((2.0 * intersection + smooth) / (np.sum(pred) + np.sum(label) + smooth)), 4)
    return dice_coefficient_score


def tree_length_calculation(pred, label_skeleton, smooth=1e-5):
    pred = pred.flatten()
    label_skeleton = label_skeleton.flatten()
    tree_length_ratio = round((np.sum(pred * label_skeleton) + smooth) / (np.sum(label_skeleton) + smooth) * 100, 2)
    return tree_length_ratio


def false_positive_rate_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()   
    label = label.flatten()
    tp = np.sum(pred * label) + smooth
    precision = round(tp * 100 / (np.sum(pred) + smooth), 3)
    return precision

def eval_metric(pred, parse, skel):
    label = tf.cast(parse > 0, tf.float32)
    pred = pred.numpy(); parse=parse.numpy(); skel=skel.numpy(); label=label.numpy()
    tp, fn, fp, iou, precision = metrics_3d(pred, label)
    dice = dice_coef(pred, label)
    total_branch_num, detected_branch_num, detected_branch_ratio = branch_detected_calculation(pred, parse, skel)
    tree_length_ration =  tree_length_calculation(pred, skel, smooth=1e-5)
    return np.array([tp, fn, fp, iou, dice, precision, total_branch_num, detected_branch_num, detected_branch_ratio, tree_length_ration])


def save_image(image, affine, name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if 'debug' in save_path:
        for i in range(image.shape[0]):
            image_nii = nib.Nifti1Image(np.float32(image[i]), affine)
            nib.save(image_nii, os.path.join(save_path, name+'_'+str(i)+'.nii.gz'))
    else:
        image_nii = nib.Nifti1Image(np.float32(image[0]), affine)
        nib.save(image_nii, os.path.join(save_path, name))
                 
def tversky(y_true, y_pred, beta=0.7, smooth = 1e-5):
    ### larger beta, larger recall
    alpha = 1 - beta
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    return (true_pos + smooth)/(true_pos + beta*false_neg + alpha*false_pos + smooth)

def tversky_loss(y_true, y_pred, beta=0.7):
    return 1 - tversky(y_true,y_pred, beta=beta)

def freeze_model(model):
    for layer in model.layers:
        layer.trainable = False
    return model

def unfreeze_model(model):
    for layer in model.layers:
        layer.trainable = True
    return model

def split_val(zipped_files, num=5, random_seed=1):
    train_files = []
    for file in zipped_files:
        train_files.append(file)
    indices = [i for i in range(len(train_files))]
    val_indices = np.random.choice(indices, num)
    val_files = [train_files[i] for i in val_indices]
    train_files = [train_files[i] for i in indices if i not in val_indices]
    return train_files, val_files


"""
import numpy as np
import tensorflow as tf
a = np.array([[0,0,3,3,3,0,0],
                [0,0,3,3,3,0,0],
                [0,0,3,3,3,0,0]], dtype=np.float32)
a = a[np.newaxis, :, :, np.newaxis]
b = tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(a)
c = tf.sign(a) * b
d = (a-c)*a
e = tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(d)
"""