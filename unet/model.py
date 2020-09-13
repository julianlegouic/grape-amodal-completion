"""
    Author: Julian LE GOUIC
    Python version: 3.6.7

    IMP Lab - Osaka Prefecture University.
"""

import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

from tensorboardX import SummaryWriter

from unet.utils import find_center
from unet.utils import prediction_figure
from unet.utils import save_classification_report
from unet.utils import save_pr_curve_plot
from unet.utils import rescale


def block_downsampling(nb_layers, c_in, c_ker):
    layers = []

    for _ in range(nb_layers):
        layers.append(nn.Conv2d(c_in, c_ker, (3, 3), stride=1, padding=1))
        layers.append(nn.ReLU())
        c_in = c_ker
    layers.append(nn.MaxPool2d((2, 2)))

    return layers


def block_upsampling(nb_layers, c_in, c_ker):
    layers = []

    layers.append(nn.ConvTranspose2d(c_in, (c_ker//2), (2, 2), stride=2))
    layers.append(nn.ReLU())
    for _ in range(nb_layers):
        layers.append(nn.Conv2d(c_in, (c_ker//2), (3, 3), stride=1, padding=1))
        layers.append(nn.ReLU())
        c_in = (c_ker//2)

    return layers


class Unet(nn.Module):

    def __init__(self, nb_classes, experiment, device,
                 c_in=1, nb_blocks=4, nb_layers=2, nb_channels=8):

        self.nb_classes = nb_classes
        self.nb_blocks = nb_blocks
        self.nb_layers = nb_layers
        self.c_in = c_in
        self.c_ker = nb_channels

        self.experiment = experiment
        self.device = device

        super(Unet, self).__init__()

        block = []
        # Downsampling
        for _ in range(self.nb_blocks):
            block += block_downsampling(self.nb_layers, self.c_in, self.c_ker)
            self.c_in = self.c_ker
            self.c_ker *= 2
        self.down = nn.Sequential(*block)

        bottom = []
        # In-between downsampling and upsampling
        for _ in range(self.nb_layers):
            bottom.append(nn.Conv2d(self.c_in, self.c_ker,
                                    (3, 3), stride=1, padding=1))
            bottom.append(nn.ReLU())
            self.c_in = self.c_ker
        self.bottom = nn.Sequential(*bottom)

        block = []
        # Upsampling
        for _ in range(self.nb_blocks):
            block += block_upsampling(self.nb_layers, self.c_in, self.c_ker)
            self.c_ker //= 2
            self.c_in = self.c_ker
        self.up = nn.Sequential(*block)

        # Last step
        self.lastConv = nn.Conv2d(
            self.c_in, self.nb_classes, (3, 3), stride=1, padding=1)

        # Resizing for targets
        self.avgPool = nn.AvgPool2d((2, 2))

        self.activation = nn.Sigmoid()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        target_shape = x.shape
        skip_connections = []

        # Downsampling
        for mod in list(self.down.modules())[1:]:
            if isinstance(mod, nn.MaxPool2d):
                skip_connections.append(x)
            x = mod(x)

        # In-between downsampling and upsampling
        for mod in list(self.bottom.modules())[1:]:
            x = mod(x)

        # Upsampling
        for mod in list(self.up.modules())[1:]:
            x = mod(x)
            if isinstance(mod, nn.ConvTranspose2d):
                last = skip_connections.pop()
                if last.shape != x.shape:
                    x = F.pad(x, (0, last.shape[-1] - x.shape[-1],
                                  last.shape[-2] - x.shape[-2], 0),
                              mode='constant', value=0)
                x = torch.cat((x, last), dim=1)

        # Last step
        x = self.lastConv(x)

        # Resizing for targets
        if x.shape != target_shape:
            x = self.avgPool(x)

        return x

    # Method used to train the model and evaluate it at each epoch
    def train_model(self, data_loader, nb_epoch, lr):
        self.train(True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        self.best_loss = np.inf

        # Initialize a writer to output logs for graphs in tensorboard
        self.writer = SummaryWriter(
            os.path.join(self.experiment, 'train', 'logs'))

        self.save_path = os.path.join(self.experiment, 'train', 'models')
        print('Models will be saved to {}/{}'.format(
            os.path.dirname(__file__), self.save_path))
        print('Start training...')
        for epoch in range(nb_epoch):
            self.current_epoch = epoch + 1
            print('Epoch n°{}'.format(self.current_epoch))
            self.train_epoch(data_loader, optimizer, criterion)

            # Save model every 1/5th of the total number of epochs
            if self.current_epoch % (nb_epoch // 5) == 0:
                print('Model saved.')
                torch.save(
                    self.state_dict(),
                    os.path.join(
                        self.save_path,
                        'model{}.pth'.format(self.current_epoch)
                    )
                )

        self.writer.export_scalars_to_json(os.path.join(
            self.experiment, 'train', 'logs', 'scalar_hist.json'))
        self.writer.close()

    def train_epoch(self, data_loader, optimizer, criterion):
        for p, phase in enumerate(['train', 'val']):
            # Loss and accuracy for the current epoch at each phase
            running_loss = 0.0
            running_accuracy = 0.0
            running_recall = 0.0
            running_precision = 0.0

            # Estimation of center of grapes
            centers_pred = np.array([])
            centers_data = np.array([])
            centers_true = np.array([])

            # Only enable gradients for training
            torch.set_grad_enabled((phase == 'train'))
            for n_batch, data in enumerate(data_loader[p]):
                # Read the data
                inputs = data['image'].to(self.device)
                target = data['target'].to(self.device)

                # Forward path + loss computing
                output = self(inputs)
                loss = criterion(output, target)
                running_loss += loss.item()

                # Zero the parameter gradients and optimize the weights
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                threshold = 0.5  # Threshold for the activation function
                scores = self.activation(output)
                pred = (scores > threshold).float()

                # Compute TP, FP, FN and FN for precision, recall and accuracy
                # When 1 pixels are equal
                tp = torch.sum((pred == 1) * (target == 1)).item()
                # Don't reward detection of berries' visible pixels
                tp -= torch.sum((pred == 1) * (inputs == 1)).item()
                # When 1 pixels are on the background
                fp = torch.sum((pred == 1) * (target == 0)).item()
                # When 0 pixels are on the berry
                fn = torch.sum((pred == 0) * (target == 1)).item()
                # When 0 pixels are on the background
                tn = torch.sum((pred == 0) * (target == 0)).item()
                running_precision += tp/(tp + fp) if (tp + fp) != 0 else 0
                running_recall += tp/(tp + fn) if (tp + fn) != 0 else 0
                running_accuracy += (tp + tn)/(tp + tn + fp + fn)

                # Retrieve centers for estimation center error
                centers_data = np.append(
                    centers_data,
                    data['image_center'].numpy()
                )
                centers_true = np.append(
                    centers_true,
                    data['target_center'].numpy()
                )
                for j in range(inputs.shape[0]):
                    cX, cY = find_center(
                        pred[j, 0].cpu().numpy().astype(np.uint8),
                        'pred'
                    )
                    centers_pred = np.append(
                        centers_pred,
                        np.array([cX, cY])
                    )
                break

            # Normalize metrics
            running_loss /= (n_batch+1)
            running_precision /= (n_batch+1)
            running_recall /= (n_batch+1)
            running_accuracy /= (n_batch+1)

            # Compute other metrics
            if (running_precision+running_recall) != 0:
                f_score = (2*running_precision*running_recall) / \
                    (running_precision+running_recall)
            else:
                f_score = 0
            pr_prec, pr_rec, _ = precision_recall_curve(
                y_true=target.view(-1).cpu().numpy(),
                probas_pred=scores.view(-1).detach().cpu().numpy()
            )
            auc_score = auc(pr_rec, pr_prec)
            ap_score = average_precision_score(
                y_true=target.view(-1).cpu().numpy(),
                y_score=pred.view(-1).cpu().numpy()
            )

            # Compute estimation centers error with L1-norm
            centers_data = centers_data.reshape(-1, 2)
            centers_true = centers_true.reshape(-1, 2)
            centers_pred = centers_pred.reshape(-1, 2)
            l1_dist_baseline = np.abs(centers_true - centers_data)
            l1_dist_baseline = np.sum(l1_dist_baseline, axis=1)
            l1_dist_baseline = np.mean(l1_dist_baseline)
            l1_dist_pred = np.abs(centers_true - centers_pred)
            l1_dist_pred = np.sum(l1_dist_pred, axis=1)
            l1_dist_pred = np.mean(l1_dist_pred)

            # Matplotlib figure of the predictions stored on tensorboard
            fig = prediction_figure(
                pred=pred,
                inputs=inputs,
                target=target,
                phase=phase,
                epoch=self.current_epoch
            )

            # Write computed metrics in tensorboard
            self.writer.add_scalar(
                '{}/accuracy'.format(phase),
                running_accuracy,
                self.current_epoch
            )
            self.writer.add_scalar(
                '{}/auc'.format(phase),
                auc_score,
                self.current_epoch
            )
            self.writer.add_scalar(
                '{}/aver_prec'.format(phase),
                ap_score,
                self.current_epoch
            )
            self.writer.add_scalar(
                '{}/l1-error-baseline'.format(phase),
                l1_dist_baseline,
                self.current_epoch
            )
            self.writer.add_scalar(
                '{}/l1-error-pred'.format(phase),
                l1_dist_pred,
                self.current_epoch
            )
            self.writer.add_scalar(
                '{}/f-score'.format(phase),
                f_score,
                self.current_epoch
            )
            self.writer.add_scalar(
                '{}/loss'.format(phase),
                running_loss,
                self.current_epoch
            )
            self.writer.add_scalar(
                '{}/precision'.format(phase),
                running_precision,
                self.current_epoch
            )
            self.writer.add_scalar(
                '{}/recall'.format(phase),
                running_recall,
                self.current_epoch
            )
            self.writer.add_pr_curve(
                '{}/pr_curve'.format(phase),
                target.view(-1),
                pred.view(-1),
                self.current_epoch
            )
            self.writer.add_figure(
                '{}/prediction'.format(phase),
                fig,
                self.current_epoch
            )

            if phase == 'val':
                # Save the model with the best loss
                if running_loss < self.best_loss:
                    self.best_loss = running_loss
                    print('Save best model.')
                    torch.save(
                        self.state_dict(),
                        os.path.join(self.save_path, 'best_model.pth')
                    )

    def predict(self, test_loader, threshold):
        self.eval()

        # Predictions and labels for the confusion matrix
        y_true = np.array([])
        y_pred = np.array([])
        # Scores for the PR-Curve
        y_score = np.array([])
        # Estimation of center of grapes
        centers_pred = np.array([])
        centers_data = np.array([])
        centers_true = np.array([])
        with torch.no_grad():
            # Solely for evaluation purpose
            if isinstance(test_loader, torch.utils.data.dataloader.DataLoader):
                print('Start evaluation...')
                for i, data in enumerate(test_loader):
                    # Read the data
                    inputs = data['image'].to(self.device)

                    # Forward path
                    output = self(inputs)
                    scores = self.activation(output)
                    pred = (scores > threshold).float()
                    target = data['target']

                    idx = ((pred == 1) * (inputs == 1))
                    preds = pred[~idx]
                    target = target[~idx]
                    scores = scores[~idx]

                    y_pred = np.append(y_pred, preds.view(-1).cpu().numpy())
                    y_true = np.append(y_true, target.view(-1).numpy())
                    y_score = np.append(y_score, scores.view(-1).cpu().numpy())

                    centers_data = np.append(
                        centers_data,
                        data['image_center'].numpy()
                    )
                    centers_true = np.append(
                        centers_true,
                        data['target_center'].numpy()
                    )
                    for j in range(inputs.shape[0]):
                        cX, cY = find_center(
                            pred[j, 0].cpu().numpy().astype(np.uint8),
                            'pred'
                        )
                        if cX != 0 and cY != 0:
                            centers_pred = np.append(
                                centers_pred,
                                np.array([cX, cY])
                            )
                        else:
                            centers_pred = np.append(
                                centers_pred,
                                data['image_center'][j].numpy()
                            )

                    if (i+1) % (len(test_loader) // 5) == 0:
                        print('Done: {}/{}'.format(i+1, len(test_loader)))
                print('Evaluation is finished. Metrics are being computed...')
                save_classification_report(
                    y_true, y_pred, threshold,
                    os.path.join(self.experiment, 'eval', 'class_rep.png')
                )
                centers_data = centers_data.reshape(-1, 2)
                centers_true = centers_true.reshape(-1, 2)
                centers_pred = centers_pred.reshape(-1, 2)
                l1_dist_baseline = np.abs(centers_true - centers_data)
                l1_dist_baseline = np.sum(l1_dist_baseline, axis=1)
                l1_dist_baseline = np.mean(l1_dist_baseline)
                l1_dist_pred = np.abs(centers_true - centers_pred)
                l1_dist_pred = np.sum(l1_dist_pred, axis=1)
                l1_dist_pred = np.mean(l1_dist_pred)
                print('Baseline center error: {}'.format(l1_dist_baseline))
                print('Center prediction error: {}'.format(l1_dist_pred))
                print('Classification report plot saved successfully.')
                save_pr_curve_plot(
                    y_true,
                    y_score,
                    os.path.join(self.experiment, 'eval', 'pr_cruve.html')
                )
                print('PR Curve plot saved successfully.')
            else:
                print('Start amodal completion...')
                # When testing in real time, not with synthetic dataset;
                # must be a tensor of shape [BxCxHxW]
                pred = torch.empty_like(test_loader)
                orig_shape = test_loader.shape
                for b, img in enumerate(test_loader):
                    img = rescale(img, (225, 325))
                    output = self(img.to(self.device))
                    res = (self.activation(output) > threshold).float()
                    res = rescale(res.squeeze().cpu(), orig_shape[2:])
                    pred[b] = res.bool()

        return pred
