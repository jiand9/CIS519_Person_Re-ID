import time

import utils
from utils import get_dataloader
import torch
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
import models
import numpy as np
from sklearn.metrics import average_precision_score


class Experiment(object):
    def __init__(self, selected_model, epoch, test_step):
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

        self.test_step = test_step
        self.selected_model = selected_model
        self.batch_size = 64
        self.train_data, self.query_data, self.gallery_data = get_dataloader(self.batch_size)
        self.num_part = 6
        self.num_class = len(self.train_data.dataset.classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = 0.1
        self.epoch = epoch
        if selected_model == 'baseline':
            self.model = models.Baseline(num_class=self.num_class)
            params = [
                {'params': self.model.backbone.parameters(), 'lr': self.lr / 10},
                {'params': self.model.local_conv.parameters(), 'lr': self.lr},
                {'params': self.model.fc.parameters(), 'lr': self.lr}
            ]
        elif selected_model == 'part-based':
            self.model = models.PartBasedCNN(num_class=self.num_class, num_part=self.num_part)
            params = [
                {'params': self.model.backbone.parameters(), 'lr': self.lr / 10},
                {'params': self.model.local_convs.parameters(), 'lr': self.lr},
                {'params': self.model.fcs.parameters(), 'lr': self.lr}
            ]
        elif selected_model == 'part-based-attention':
            self.model = models.PartBasedAttentionCNN(num_class=self.num_class, num_part=self.num_part)
            params = [
                {'params': self.model.backbone.parameters(), 'lr': self.lr / 10},
                {'params': self.model.local_convs.parameters(), 'lr': self.lr},
                {'params': self.model.fcs.parameters(), 'lr': self.lr}
            ]
        else:
            raise Exception('Invalid model type')

        self.model = self.model.to(self.device)
        self.optimizer = optim.SGD(params=params, momentum=0.9, weight_decay=5e-4, nesterov=True)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()

        self.track_loss = []
        self.track_CMC = []
        self.track_mAP = []

    def train(self):
        for epoch in range(self.epoch):
            self.model.train()

            epoch_loss = 0
            for inputs, labels in self.train_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                if self.selected_model == 'baseline':
                    loss = self.criterion(outputs, labels)
                else:
                    loss = 0
                    for output in outputs:
                        loss += self.criterion(output, labels)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * inputs.shape[0]

            epoch_loss = epoch_loss / len(self.train_data.dataset.imgs)
            self.track_loss.append(epoch_loss)

            print("time:{}, epoch:{}, train loss:{:.4f}".format(int(time.time()), epoch, epoch_loss))

            if (epoch + 1) % self.test_step == 0:
                CMC, mAP = self.test()
                self.track_CMC.append(CMC)
                self.track_mAP.append(mAP)
                print(
                    "time:{}, rank-1:{:.4f}, rank-5:{:.4f}, rank-10:{:.4f}, mAP:{:.4f}".format(int(time.time()), CMC[0],
                                                                                               CMC[4], CMC[9], mAP))

            self.scheduler.step()

    def _extract_features(self, dataloader):
        features = []
        for inputs, _ in dataloader:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model.extract_feature(inputs)

            norm = outputs.norm(p=2, dim=1)
            outputs = outputs.div(norm.unsqueeze(dim=1))

            features.append(outputs)

        features = torch.cat(features, dim=0)

        return features

    def test(self):
        self.model.eval()
        gallery_cameras, gallery_labels = utils.get_camera_label(self.gallery_data.dataset.imgs)
        query_cameras, query_labels = utils.get_camera_label(self.query_data.dataset.imgs)

        gallery_features = self._extract_features(self.gallery_data)
        query_features = self._extract_features(self.query_data)

        CMC = np.zeros(len(gallery_labels))
        AP = 0

        for i in range(len(query_labels)):
            query_feature = query_features[i]
            query_label = query_labels[i]
            query_camera = query_cameras[i]

            f = query_feature.shape[0]
            scores = torch.mm(gallery_features, query_feature.view(f, -1)).cpu().data.numpy().flatten()
            match = np.argwhere(gallery_labels == query_label)
            same_camera = np.argwhere(gallery_cameras == query_camera)

            positive = np.setdiff1d(match, same_camera)

            junk = np.intersect1d(match, same_camera)
            index = np.arange(len(gallery_labels))
            valid = np.setdiff1d(index, junk)

            y_true = np.in1d(valid, positive)
            y_score = scores[valid]
            AP += average_precision_score(y_true, y_score)

            sorted_score_index = np.argsort(y_score)[::-1]
            sorted_y_true = y_true[sorted_score_index]

            true_index = np.argwhere(sorted_y_true == True).flatten()

            if len(true_index) > 0:
                CMC[true_index[0]:] += 1

        CMC = CMC / len(query_labels) * 100
        mAP = AP / len(query_labels) * 100

        return CMC, mAP
