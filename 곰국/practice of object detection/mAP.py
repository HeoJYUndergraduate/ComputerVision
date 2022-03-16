import torch
from NonMaxSuppression import non_max_suppression
from iou import intersection_over_union
from collections import Counter


def mean_average_precision( ##iou threshold는 class마다 달라야 한다
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=20
):

    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes  
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]  
        true_boxes (list): Similar as pred_boxes except all the correct ones  
        iou_threshold (float): threshold where predicted bboxes is correct  
        box_format (str): "midpoint" or "corners" used to specify bboxes  
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """
    #pred_boxes : [[train_index, class_prediciton, prob_score, x1, y1, x2, y2],[],[]]
    average_precisions=[]
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths= []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        for true_box in true_boxes:
            if true_box[1] ==c:
                ground_truths.append(true_box)                    
        #img 0 has 3 bboxes
        #img 1 has 5 bboxes
        #amount_boxes  =  {0:3, 1:5}      
        amount_bboxes = Counter(gt[0] for gt in ground_truths)      

        for key, val in amount_bboxes.item:
            amount_bboxes[key] = torch.zeros(val)   #내용이 숫자 -----> n개의 0이 있는 tensor로 변환
        detections.sort(key = lambda x:x[2], reverse = True)    # prob_score 이 높은 순서대로
        TP= torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        for detections_idx, detection in enumerate(detections):
            ground_truth_img=[
                bbox for bbox in ground_truths if bbox[0] == detection[0]    # detection과 index가 같은 bounding boxes
            ]
            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx_gt, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]),torch.tensor(gt[3:]),box_format = box_format)

                if iou > best_iou:
                    best_iou = iou 
                    best_gt_idx = idx_gt

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detections_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detections_idx] = 1
            else:
                FP[detections_idx] =1         


        #[1,1,0,1,0] -> [1,2,2,3,3]
        TP_cumsum = torch.cumsum(TP, dim = 0)
        FP_cumsum = torch.cumsum(FP, dim = 0)    
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + +FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precisions.append(torch.trapz(precisions, recalls))   # 넓이 계산

    #average_precisions: class 별로 Area of Precision이 저장되어 있음
    return sum(average_precisions)/len(average_precisions)