"""
    Author: Julian LE GOUIC
    Python version: 3.6.7

    IMP Lab - Osaka Prefecture University.
"""

import argparse
import cv2
import itertools as itrt
import glob
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go
import plotly.io as pio
import time
import torch
import torchvision.transforms as tf

from matplotlib.legend import Legend
from skimage import io
from skimage.morphology import convex_hull_image
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics import classification_report

# matplotlib.use('Agg')  # Comment if you want to plot while debugging


def prediction_figure(pred, inputs, target, phase, epoch):
    """
    Return the figure of the prediction and pixel-labeled prediction
    to be displayed in tensorboard.

    Args:
        pred (tensor): Tensor of all predictions of a batch.
        inputs (tensor): Corresponding input tensor.
        target (tensor): Corresponging target tensor.
        phase (string): String of the current phase (train
            or val).
        epoch (int): Number of the current epoch.
    """

    # Clear previous figures
    plt.cla()
    plt.clf()
    plt.close()

    phase = 'training' if phase == 'train' else 'validation'

    # Create dictionnaries of labels for pixels of pred and res
    raw_lbl = {
        'Background': 0,
        'Berry': 1
    }

    labels = {
        'True Negative': 0,
        'False Negative': 1,
        'False Positive': 2,
        'Known pixels': 3,
        'True Positive': 4
    }

    # Choose a random sample from the batch
    idx = np.random.randint(0, pred.shape[0])

    # Create subplots figure
    fig, (ax_raw, ax_lbl) = plt.subplots(1, 2, figsize=(10, 7))

    # Create figure for raw prediction
    raw = ax_raw.imshow(pred[idx, 0].cpu().numpy(), plt.get_cmap('inferno'))

    # Create normalized array from the values of raw_lbl dictionnary
    raw_col = np.fromiter(raw_lbl.values(), dtype=int) / max(raw_lbl.values())
    # Get the colors of the values, according to the colormap used by imshow
    colors_raw = [raw.cmap(col) for col in raw_col]
    # Create a patch (proxy artist) for every color
    patches_raw = [mpatches.Patch(color=colors_raw[j], label='{}'.format(
        key)) for j, key in enumerate(raw_lbl.keys())]

    # Create tensor for pixel-labeled image of the prediction
    res = torch.empty_like(pred, device='cpu')
    res[(pred == 0) * (target == 0)] = labels['True Negative']
    res[(pred == 0) * (target == 1)] = labels['False Negative']
    res[(pred == 1) * (target == 0)] = labels['False Positive']
    res[(pred == 1) * (target == 1)] = labels['True Positive']
    # Already known pixels (visible pixels)
    # Operation done in last to overwrite TP pixels
    res[(pred == 1) * (inputs == 1)] = labels['Known pixels']

    # Create figure for pixel-labeled image of the prediction
    lbl = ax_lbl.imshow(res[idx, 0].numpy(), cmap=plt.get_cmap('inferno'))

    # Create colorscale and patches for legend of pixel-labeled image
    lbl_col = np.fromiter(labels.values(), dtype=int) / max(labels.values())
    colors_lbl = [lbl.cmap(col) for col in lbl_col]
    patches_lbl = [mpatches.Patch(color=colors_lbl[j], label='{}'.format(
        key)) for j, key in enumerate(labels.keys())]

    # Figures properties and legends
    ax_raw.set_title('Raw prediction')
    raw_leg = Legend(
        parent=ax_lbl,
        handles=patches_raw,
        labels=raw_lbl.keys(),
        title='Raw prediction',
        loc=2, bbox_to_anchor=(1.05, 1),
        frameon=True, fancybox=True, shadow=True,
        borderaxespad=0.0
    )
    ax_raw.tick_params(
        axis='both',
        left=False, right=False,
        top=False, bottom=False,
        labelleft=False, labelright=False,
        labeltop=False, labelbottom=False
    )

    ax_lbl.set_title('Pixel-labeled prediction')
    ax_lbl.legend(
        handles=patches_lbl,
        title='Pixel-labeled prediction',
        loc=2, bbox_to_anchor=(1.05, 0.85),
        frameon=True, fancybox=True, shadow=True,
        borderaxespad=0.0
    )
    ax_lbl.add_artist(raw_leg)
    ax_lbl.tick_params(
        axis='both',
        left=False, right=False,
        top=False, bottom=False,
        labelleft=False, labelright=False,
        labeltop=False, labelbottom=False
    )

    fig.suptitle('Prediction: end of epoch {} ({} step)'.format(epoch, phase))
    plt.tight_layout()

    return fig


def save_classification_report(y_true, y_pred, threshold, path):
    """
    Save the plot of the classification report from scikit-learn built-in
    function.

    Args:
        y_true (array): True binary labels.
        y_pred (array): Predictions of the model.
        threshold (float): Threshold used during the evaluation.
        path (string): Path where to save the plot of the classification
            report matrix.
    """

    classificationReport = classification_report(
        y_true, y_pred,
        target_names=['Background', 'Berry'], output_dict=True
    )

    # Retrieve the lines of the CR and the class names
    lines = list(classificationReport.keys())
    class_names = lines[:-3]

    # Retrieve informations from the dictionary
    pltMat, support, classes, num_pred = [], [], [], []
    for line in lines:
        if line == 'accuracy':
            gaccuracy = float(str(classificationReport[line])[:4])
        else:
            tp = classificationReport[line]['recall'] * \
                classificationReport[line]['support']
            if classificationReport[line]['precision'] != 0:
                num_pred.append(
                    np.round(tp / classificationReport[line]['precision']))
            else:
                num_pred.append(0)
            pltMat.append([
                classificationReport[line]['precision'],
                classificationReport[line]['recall'],
                classificationReport[line]['f1-score']
            ])
            support.append(classificationReport[line]['support'])
            classes.append(line)

    # Convert pltMat to numpy array and normalize the accuracy column
    pltMat = np.array(pltMat)

    # Set labels for axis
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(classes[idx], sup)
                   for idx, sup in enumerate(support)]

    # Plot the report matrix
    plt.imshow(
        pltMat,
        cmap='RdBu', vmin=0, vmax=1,
        aspect='auto', interpolation='nearest'
    )

    plt.title('Classification report: accuracy={0} at t={1:.2f}'.format(
        gaccuracy, threshold))
    plt.colorbar()
    plt.xticks(np.arange(len(xticklabels)), xticklabels, rotation=45)
    plt.yticks(np.arange(len(yticklabels)), yticklabels)

    # Set threshold for print color text
    upper_thresh = 0.8
    lower_thresh = 0.2

    # Print text on each cells
    def color_txt(x):
        return 'white' if (x > upper_thresh or x < lower_thresh) else 'black'
    for i, j in itrt.product(range(pltMat.shape[0]), range(pltMat.shape[1])):
        if (i < len(class_names) and j == 0):
            t = format(pltMat[i, j], '.2f')+'\n({})'.format(int(num_pred[i]))
        else:
            t = format(pltMat[i, j], '.2f')
        plt.text(j, i, t,
                 horizontalalignment='center',
                 verticalalignment='center',
                 color=color_txt(pltMat[i, j]))

    plt.xlabel('Metrics')
    plt.ylabel('')

    plt.tight_layout()
    plt.savefig(path)
    plt.close('all')


def save_pr_curve_plot(y_true, y_scores, path):
    """
    Save the plot of the PR-Cruve for the evaluation of the model.

    Args:
        y_true (array): True binary labels.
        y_score (array): Estimated probabilities from the model.
        path (string): Path where to save the plot of the PR-Curve.
    """

    # No skill model (all predictions in negative/majority class)
    no_skill_probs = [0 for _ in range(len(y_true))]

    # Compute precision/recall for no-skill model and our model
    ns_prec, ns_rec, _ = precision_recall_curve(y_true, no_skill_probs)
    mod_prec, mod_rec, thresholds = precision_recall_curve(y_true, y_scores)
    thresholds = np.append(thresholds, 1)

    mod_auc = auc(mod_rec, mod_prec)  # Compute AUC

    hover_temp = 'Recall: %{x}<br>' + \
        'Precision: %{y}<br>' + 'Threshold: %{text}'

    # Create figure
    fig = go.Figure()

    # Plot no-skill model's and our model's PR-curves
    fig.add_trace(
        go.Scatter(
            x=ns_rec,
            y=ns_prec,
            name='No skill',
            mode='lines',
            line=dict(
                color='blue',
                width=1,
                dash='dot'
            ),
            hoverinfo='skip'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mod_rec,
            y=mod_prec,
            name='Model',
            line=dict(
                color='orange',
                width=2
            ),
            text=thresholds,
            hovertemplate=hover_temp
        )
    )

    fig.update_layout(
        title=dict(
            text='Precision-Recall curve: AUC={0:.2f}'.format(mod_auc),
            font=dict(
                size=28
            ),
            xanchor='center',
            yanchor='top'
        ),
        xaxis_title='Recall',
        yaxis_title='Precision',
        template='seaborn'
    )

    # Save figure as html file
    pio.write_html(fig=fig, file=path, auto_open=False)


def find_center(mask, key):
    """
    Find the centroid of the binary mask.

    Args:
        mask (array): Binary array mask.
        key (string): Key to access the sample elements. Also used
            in training for prediction.
    """

    mask = np.uint8(mask > 0)
    if key == 'image' or key == 'target':
        if not is_predictable(mask):
            return 0, 0
        else:
            mask = convex_hull_image(mask).astype(np.uint8)
            output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    elif key == 'pred':
        try:
            if not is_predictable(mask):
                return 0, 0
            output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
            if len(output[3]) != 2:
                # only two centroids: one for background and one for grape
                raise IndexError('Could not find centroids of one grape.')
        except IndexError:
            mask = convex_hull_image(mask).astype(np.uint8)
            if not is_predictable(mask):
                return 0, 0
            output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
            if len(output[3]) != 2:
                return 0, 0

    # centroids of grape class is always in index 1 (last)
    coords = output[3][-1]
    if np.isnan(coords[0]) or np.isnan(coords[1]):
        return 0, 0
    x_f, y_f = int(np.floor(coords[0])), int(np.floor(coords[1]))
    x_c, y_c = int(np.ceil(coords[0])), int(np.ceil(coords[1]))

    if mask[y_f, x_f] == 1:
        x, y = x_f, y_f
    elif mask[y_c, x_c] == 1:
        x, y = x_c, y_c
    elif mask[y_c, x_f] == 1:
        x, y = x_f, y_c
    elif mask[y_f, x_c] == 1:
        x, y = x_c, y_c
    else:
        x, y = 0, 0

    return x, y


def is_predictable(mask_pred):
    """
    Returns a boolean to indicates if we came estimate the center from
    the prediction mask directly.

    Args:
        mask_pred (array): Binary array mask of the prediction.
    """

    # if the mask_pred is full of 0
    black = np.sum(mask_pred) == 0

    # if the mask_pred contains mor 1s than 0s
    unique, counts = np.unique(mask_pred, return_counts=True)
    if len(unique) == 1:
        ones = True if unique[0] == 1 else False
    else:
        ones = counts[1] > counts[0]
    return not(black and ones)


def rescale(img, shape, t=0.5):
    """ Rescale images when doing amodal completion.

    Args:
        img (Tensor): Tensor to be resized.
        shape (tuple): Shape (h, w) desired for the output.
    """

    img = tf.ToPILImage()(img)
    img = tf.Resize(shape, 3)(img)
    img = (np.array(img) > t*255).astype(np.uint8)*255
    img = tf.ToTensor()(img).unsqueeze(0)
    return img


def number_min_epoch(arg):
    """
    Type function for argparse - a minimum of 5 epochs for training.

    Args:
        arg (string): Number of epoch argument parsed from ArgumentParser().
    """

    if int(arg) < 5:
        raise argparse.ArgumentTypeError('Argument for epoch must be >= 5.')
    return int(arg)


def my_collate(batch):
    """
    Collate function for Pytorch DataLoader object. Used to ignore difficult
    images (i.e. discard outlier berries).

    Args:
        batch (list): List of items retrieved from __getitem__ method of
            the dataset.
    """

    image, target, color = [], [], []
    image_center, target_center = [], []
    for elem in batch:
        image.append(elem['image'])
        target.append(elem['target'])
        color.append(torch.from_numpy(elem['color']))
        image_center.append(torch.from_numpy(elem['image_center']))
        target_center.append(torch.from_numpy(elem['target_center']))
    image = torch.stack(image, dim=0)
    target = torch.stack(target, dim=0)
    color = torch.stack(color, dim=0)
    image_center = torch.stack(image_center, dim=0)
    target_center = torch.stack(target_center, dim=0)
    batch = {'image': image, 'target': target, 'color': color,
             'image_center': image_center, 'target_center': target_center}
    return batch


def check_plot(mask, a_mask, target):
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 10))

    ax[0].imshow(mask.numpy())
    ax[1].imshow(a_mask.numpy())
    ax[2].imshow(target.numpy())

    plt.tight_layout()
    plt.show()


def f_score(a, b):
    return (2*a*b)/(a+b) if (a+b) != 0 else 0


def find_gt_image(a_mask, gt_list):
    best_precision = 0
    best_recall = 0
    best_fscore = 0
    best_accuracy = 0
    best_target = None
    for gt_name in gt_list:
        # Read image and convert it into 2D binary tensor
        target = io.imread(gt_name)[:, :, :3]  # discard alpha channel
        grape_color = np.array([
            np.max(target[:, :, 0]),
            np.max(target[:, :, 1]),
            np.max(target[:, :, 2])
        ]).reshape(1, 1, 3)
        col_mask = (target == grape_color).astype(int)
        target = (col_mask.sum(-1) == 3).astype(np.uint8)*255
        target = tf.ToTensor()(target).squeeze()

        # Precision, recall, accuracy for amodal masks
        tp = torch.sum((a_mask == 1) * (target == 1)).item()
        fp = torch.sum((a_mask == 1) * (target == 0)).item()
        fn = torch.sum((a_mask == 0) * (target == 1)).item()
        tn = torch.sum((a_mask == 0) * (target == 0)).item()
        tmp_precision = tp/(tp + fp) if (tp + fp) != 0 else 0
        tmp_recall = tp/(tp + fn) if (tp + fn) != 0 else 0
        tmp_fscore = f_score(tmp_precision, tmp_recall)
        tmp_accuracy = (tp + tn)/(tp + tn + fp + fn)

        if tmp_fscore > best_fscore:
            best_precision = tmp_precision
            best_recall = tmp_recall
            best_fscore = tmp_fscore
            best_accuracy = tmp_accuracy
            best_target = target

    return best_precision, best_recall, best_fscore, best_accuracy, best_target


def evaluate_amodal(masks, amodal_masks, caption):
    # Path to amodal ground-truth + reg exp to isolate gt images of caption
    path = os.path.join('data', 'fastgrape', 'gtgrape',
                        'amodal', '{}*'.format(caption[:-4]))

    gt_lists = [
        sorted(glob.glob(os.path.join(os.getcwd(), path))),
        sorted(glob.glob(os.path.join(os.getcwd(), path)), reverse=True)
    ]

    best_matching_fscore = 0
    best_results = None
    wrong_masks = None

    for gt_list in gt_lists:
        # Initialiaze metrics variables
        centers_modal = np.array([])
        centers_amodal = np.array([])
        centers_gt = np.array([])
        modal_running_precision = 0
        modal_running_recall = 0
        modal_running_fscore = 0
        modal_running_accuracy = 0
        amodal_running_precision = 0
        amodal_running_recall = 0
        amodal_running_fscore = 0
        amodal_running_accuracy = 0

        centers_time = 0
        gt_image_time = 0

        wrong_masks_idx = []
        for idx, (mask, a_mask) in enumerate(zip(masks, amodal_masks)):
            # print('Mask no:', idx)

            t1 = time.time()
            # Find the corresponding gt amodal image to the current amodal
            # masks by taking the image with the best precision score
            best_precision = 0
            best_recall = 0
            best_fscore = 0
            best_accuracy = 0
            best_target = None
            for gt_name in gt_list:
                # Read image and convert it into 2D binary tensor
                target = io.imread(gt_name)[:, :, :3]  # discard alpha channel
                grape_color = np.array([
                    np.max(target[:, :, 0]),
                    np.max(target[:, :, 1]),
                    np.max(target[:, :, 2])
                ]).reshape(1, 1, 3)
                col_mask = (target == grape_color).astype(int)
                target = (col_mask.sum(-1) == 3).astype(np.uint8)*255
                target = tf.ToTensor()(target).squeeze()

                # Precision, recall, accuracy for amodal masks
                tp = torch.sum((a_mask == 1) * (target == 1)).item()
                fp = torch.sum((a_mask == 1) * (target == 0)).item()
                fn = torch.sum((a_mask == 0) * (target == 1)).item()
                tn = torch.sum((a_mask == 0) * (target == 0)).item()
                tmp_precision = tp/(tp + fp) if (tp + fp) != 0 else 0
                tmp_recall = tp/(tp + fn) if (tp + fn) != 0 else 0
                tmp_fscore = f_score(tmp_precision, tmp_recall)
                tmp_accuracy = (tp + tn)/(tp + tn + fp + fn)

                if tmp_fscore > best_fscore:
                    best_precision = tmp_precision
                    best_recall = tmp_recall
                    best_fscore = tmp_fscore
                    best_accuracy = tmp_accuracy
                    best_target = target
                    target_name = gt_name

            t2 = time.time()
            gt_image_time += t2 - t1

            # results = find_gt_image(a_mask, gt_list)
            if best_target is None:
                # pudb.set_trace()
                # results = find_gt_image(a_mask, gt_list)
                wrong_masks_idx.append(idx)
                continue

            # Remove already matched gt image
            gt_list.remove(target_name)

            # Update metrics for amodal mask
            amodal_running_precision += best_precision
            amodal_running_recall += best_recall
            amodal_running_fscore += best_fscore
            amodal_running_accuracy += best_accuracy

            # Retrieve centers from modal, amodal and gt masks
            cX, cY = find_center(
                mask.numpy().astype(np.uint8),
                'pred'
            )
            centers_modal = np.append(
                centers_modal,
                np.array([cX, cY])
            )

            cX, cY = find_center(
                a_mask.numpy().astype(np.uint8),
                'pred'
            )
            centers_amodal = np.append(
                centers_amodal,
                np.array([cX, cY])
            )

            cX, cY = find_center(
                best_target.numpy().astype(np.uint8),
                'pred'
            )
            centers_gt = np.append(
                centers_gt,
                np.array([cX, cY])
            )
            t3 = time.time()
            centers_time += t3 - t2

            # Precision, recall, accuracy for modal masks
            tp = torch.sum((mask == 1) * (best_target == 1)).item()
            fp = torch.sum((mask == 1) * (best_target == 0)).item()
            fn = torch.sum((mask == 0) * (best_target == 1)).item()
            tn = torch.sum((mask == 0) * (best_target == 0)).item()
            modal_running_precision += tp/(tp + fp) if (tp + fp) != 0 else 0
            modal_running_recall += tp/(tp + fn) if (tp + fn) != 0 else 0
            modal_running_fscore = f_score(
                modal_running_precision, modal_running_recall)
            modal_running_accuracy += (tp + tn)/(tp + tn + fp + fn)

        # Normalize metrics
        idx -= len(wrong_masks_idx)
        modal_running_precision /= (idx+1)
        modal_running_recall /= (idx+1)
        modal_running_fscore /= (idx+1)
        modal_running_accuracy /= (idx+1)
        amodal_running_precision /= (idx+1)
        amodal_running_recall /= (idx+1)
        amodal_running_fscore /= (idx+1)
        amodal_running_accuracy /= (idx+1)

        centers_time /= (idx+1)
        gt_image_time /= (idx+1)

        # print('Time to find centers: {}s'.format(centers_time))
        # print('Time to find gt image: {}s'.format(gt_image_time))

        # Compute error of center estimations
        centers_modal = centers_modal.reshape(-1, 2)
        centers_gt = centers_gt.reshape(-1, 2)
        centers_amodal = centers_amodal.reshape(-1, 2)
        l1_dist_modal = np.abs(centers_gt - centers_modal)
        l1_dist_modal = np.sum(l1_dist_modal, axis=1)
        l1_dist_modal = np.mean(l1_dist_modal)
        l1_dist_amodal = np.abs(centers_gt - centers_amodal)
        l1_dist_amodal = np.sum(l1_dist_amodal, axis=1)
        l1_dist_amodal = np.mean(l1_dist_amodal)
        # print('Compute estimation error: {}s'.format(time.time() - t3))

        results = {
            'modal': {
                'precision': modal_running_precision,
                'recall': modal_running_recall,
                'f-score': modal_running_fscore,
                'accuracy': modal_running_accuracy,
                'center_err': l1_dist_modal
            },
            'amodal': {
                'precision': amodal_running_precision,
                'recall': amodal_running_recall,
                'f-score': amodal_running_fscore,
                'accuracy': amodal_running_accuracy,
                'center_err': l1_dist_amodal
            }
        }

        if results['amodal']['f-score'] > best_matching_fscore:
            best_matching_fscore = results['amodal']['f-score']
            best_results = results
            wrong_masks = wrong_masks_idx

    return best_results, wrong_masks
