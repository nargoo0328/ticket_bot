import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import os

class Captcha:
    def __init__(self,n,model,loss=None):
        self.n = n
        self.model = model
        self.metric = captcha_metric(n)

        if loss is not None:
            self.loss = loss
        else:
            self.loss = nn.CrossEntropyLoss()

    def compute_loss(self,pred,target):
        """
            nn.CrossEntropyLoss: 
                pred: NxC
                target: N
        """

        pred = pred.reshape(-1,27)
        target = target.reshape(-1)
        loss = self.loss(pred,target)
        return loss
    
    def run(self,data):
        label = data['label']
        pred = self.model(data['image'])
        loss = self.compute_loss(pred,label)
        self.metric.update(pred,label)

        return pred, loss

    def demo(self,pred,data):
        file_name, label = data['file_name'], data['label']
        b = pred.shape[0]
        for i in range(b):
            pred_b = pred[i]
            pred_b = F.softmax(pred_b,-1).argmax(-1) # self.n
            output_pred, output_gt = '', ''
            for s1,s2 in zip(pred_b,label[i]):
                if s1 == 0:
                    continue
                output_pred += chr(96+s1)
                output_gt += chr(96+s2)
            
            print(f"Prediction: {output_pred} gt: {output_gt}")
            img = cv2.imread(file_name[i],cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(128,96),interpolation=cv2.INTER_NEAREST)
            cv2.imshow('CAPTCHA',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

class captcha_metric:
    def __init__(self,n):
        self.n = n
        self.tp = None
        self.total = None
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.total = 0.0

    def update(self,pred,label):
        b = pred.shape[0]
        pred = F.softmax(pred,-1).argmax(-1) # b,n
        tp = pred == label
        count = tp.sum(1)
        self.tp += (count == self.n).sum()
        self.total += b

    def compute(self):
        acc = self.tp/self.total
        return acc 