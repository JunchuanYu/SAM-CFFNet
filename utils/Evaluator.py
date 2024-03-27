import numpy as np
import torch

np.seterr(divide='ignore',invalid='ignore')

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def OverallAccuracy(self):  
        #  返回所有类的整体像素精度OA
        # acc = (TP + TN) / (TP + TN + FP + TN)  
        OA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()  
        return OA
    
    def Precision(self):  
        #  返回所有类别的精确率precision  
        precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis = 0)
        return precision  

    def Recall(self):
        #  返回所有类别的召回率recall
        recall = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis = 1)
        return recall
    
    def F1Score(self):
        precision = self.Precision()
        recall = self.Recall()
        f1score = 2 * precision * recall / (precision + recall)
        return f1score

    def IntersectionOverUnion(self):  
        #  返回交并比IoU
        intersection = np.diag(self.confusion_matrix)  
        union = np.sum(self.confusion_matrix, axis = 1) + np.sum(self.confusion_matrix, axis = 0) - np.diag(self.confusion_matrix)  
        IoU = intersection / union
        return IoU

    def MeanIntersectionOverUnion(self):  
        #  返回平均交并比mIoU
        intersection = np.diag(self.confusion_matrix)  
        union = np.sum(self.confusion_matrix, axis = 1) + np.sum(self.confusion_matrix, axis = 0) - np.diag(self.confusion_matrix)  
        IoU = intersection / union
        mIoU = np.nanmean(IoU)  
        return mIoU
    
    def Frequency_Weighted_Intersection_over_Union(self):
        #  返回频权交并比FWIoU
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
    
    def _generate_matrix(self, gt_image, pre_image):
        if  'torch' in str(gt_image.dtype):
            gt_image = gt_image.cpu()
            gt_image = gt_image.numpy()
        if 'torch' in str(pre_image.dtype):
            pre_image = pre_image.cpu()
            pre_image = pre_image.numpy()
        gt_image = gt_image.astype('int') 
        pre_image = pre_image.astype('int')
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask]+ pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)