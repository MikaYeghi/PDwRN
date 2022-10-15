from pathlib import Path
import os
import pickle
import torch
from collections import OrderedDict

from detectron2.evaluation import DatasetEvaluator

import pdb

class PDwRNEvaluator(DatasetEvaluator):
    def __init__(self, cfg, test_data_path):
        super(PDwRNEvaluator, self).__init__()
        self.threshold = cfg.MODEL.RETINANET.SCORE_THRESH_TEST # threshold which is applied to predictions
        self.data_path = test_data_path
        
        # Counters (there is no true negative in this scenario)
        self.TP = 0 # true positive
        self.FN = 0 # false negative
        self.FP = 0 # false positive
    
    def reset(self):
        self.TP = 0
        self.FN = 0
        self.FP = 0
    
    def process(self, inputs, outputs):
        TP_count = 0
        FN_count = 0
        FP_count = 0
        for input_, output_ in zip(inputs, outputs):
            # Extract the ground-truth labels using the input
            filename = Path(input_['file_name']).stem
            with open(os.path.join(self.data_path, "annotations", filename + '.pkl'), 'rb') as f:
                annotations = pickle.load(f)
            del annotations['unknown'] # delete the unknown vehicles from the dataset
            gt_points = []
            for vehicle_type in annotations.keys():
                for vehicle_coordinate in annotations[vehicle_type]:
                    vehicle = {
                        "gt_point": torch.tensor(vehicle_coordinate).long().cuda(),
                        "category_id": torch.tensor(0).long().cuda(),
                        "status": torch.tensor(0).long().cuda()
                    }
                    gt_points.append(vehicle) # appends a numpy array consisting of 2 values in (x, y) format
            
            # Instantiate the list of predictions
            pred_points = {
                "points": output_['instances'].pred_boxes.get_centers().long().cuda(),
                "status": torch.tensor([0 for _ in range(len(output_['instances']))]).long().cuda()
            }
            
            # Evaluate the predictions based on the ground-truth points
            for gt_idx in range(len(gt_points)):
                gt_point = gt_points[gt_idx]
                for pred_idx in range(len(pred_points['points'])):
                    pred_point = pred_points['points'][pred_idx]
                    
                    # Compute the distance between the given pair of points
                    # NOTE: a prediction is accepted if it's within 6 pixels from the ground-truth point
                    # TO DO: exclude predictions and GT-points which are 6 pixels close to any of the image borders
                    gt_pred_dist = (gt_point['gt_point'] - pred_point).pow(2).sum().sqrt()
                    if gt_pred_dist < 6:
                        gt_points[gt_idx]['status'] = torch.tensor(1).long().cuda()
                        pred_points['status'][pred_idx] = torch.tensor(1).long().cuda()
                        break
                if gt_point['status'] == 1:
                    TP_count += 1
                else:
                    FN_count += 1
            # Each 0 in prediction statuses represents a false positive
            if len(pred_points['status']) > 0:
                FP_count += torch.unique(pred_points['status'], return_counts=True)[1][0].item()
        
        # Add the counts to the global counters
        self.TP += TP_count
        self.FN += FN_count
        self.FP += FP_count
    
    def evaluate(self):
        results = OrderedDict()
        # results["precision"] = self.TP / (self.TP + self.FP)
        # results["recall"] = self.TP / (self.TP + self.FN)
        # results["FP_rate"] = self.FP / (self.FP + self.TP)
        
        # Avoid division by 0
        if self.TP + self.FP > 0:
            results["precision"] = self.TP / (self.TP + self.FP)
        else:
            results["precision"] = 0.0
        if self.TP + self.FN > 0:
            results["recall"] = self.TP / (self.TP + self.FN)
        else:
            results["recall"] = 0.0
        
        return results