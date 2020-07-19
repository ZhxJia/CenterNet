# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
# from .utils import _transpose_and_gather_feat
import torch.nn.functional as F


def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1)
  neg_inds = gt.lt(1)

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _generalized_iou(pred, gt):
  '''
  Input:
      pred: (x1,y1,x2,y2)
      gt: (x1,y1,x2,y2)
  Output:
      GIOU = IOU - C\(AUB) / C
  '''
  assert pred.shape == gt.shape

  lt = torch.max(pred[:,:2], gt[:,:2])
  rb = torch.min(pred[:,2:], gt[:,2:])
  wh = (rb - lt).clamp(min=0) # overlap
  enclose_x1y1 = torch.min(pred[:, :2], gt[:, :2])
  enclose_x2y2 = torch.max(pred[:, 2:], gt[:, 2:])
  enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0) # C

  overlap = wh[:, 0] * wh[:, 1]
  ap = (pred[:,2] - pred[:,0]) * (pred[:,3] - pred[:,1])
  ag = (gt[:,2] - gt[:,0]) * (gt[:,3] - gt[:,1])
  ious = overlap / (ap + ag - overlap + 1e-6)

  enclose_area = enclose_wh[:,0] * enclose_wh[:,1] # C
  u = ap + ag - overlap
  gious = ious - (enclose_area-u) / (enclose_area + 1e-6)
  giou_metric = 1.0 - gious
  return giou_metric


def _slow_reg_loss(regr, gt_regr, mask):
  num  = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr)

  regr    = regr[mask]
  gt_regr = gt_regr[mask]
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegGiouLoss(nn.Module):
  '''
  https://arxiv.org/pdf/1902.09630.pdf
  Input:
    pred_hm -> [batch, 80, h, w] tensor
    pred_wh -> [batch, 4, h, w] tensor  (wl, ht, wr, hb) within heatmap
    target_hm -> [batch, 80, h, w] tesnor
    target_box -> [batch, 4, h, w] tesnor (x1, y1, x2, y2) within heatmap
    weight -> [batch, 1, h, w] tensor
  Output:
    giou loss -> [batch,1]  tensor
  Note:
    In paper: TTFNet, the gious loss calculate on raw image scale
  Value range:
    approximate: <2.
  '''
  def __init__(self):
    super(RegGiouLoss,self).__init__()
    self.base_loc = None

  def forward(self, pred_hm, pred_wh, target_hm,target_boxes ,weight, avg_factor=None):
    H, W = pred_hm.shape[2:]

    mask = weight.view(-1, H, W)
    avg_factor = mask.sum() + 1e-4

    if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
      loc_x = torch.arange(0, W, dtype=torch.float32, device=pred_wh.device)
      loc_y = torch.arange(0, H, dtype=torch.float32, device=pred_wh.device)
      loc_y, loc_x = torch.meshgrid(loc_y, loc_x)
      self.base_loc = torch.stack((loc_y, loc_x), dim=0) # (2,H,W)

    pred_boxes = torch.cat((self.base_loc - pred_wh[:, [0, 1]],
                            self.base_loc + pred_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1) # （batch,h,w,4）
    target_boxes = target_boxes.permute(0, 2, 3, 1)  # [batch, h, w, 4]
    #TODO debug:
    pos_mask = mask > 0 #[batch, 128, 128]
    mask = mask[pos_mask].float()

    pd_boxes = pred_boxes[pos_mask].view(-1,4) #[num_boxes, 4]
    gt_boxes = target_boxes[pos_mask].view(-1,4) #[num_boxes, 4]
    giou = _generalized_iou(pd_boxes,gt_boxes)
    loss = torch.sum(giou * mask)[None] / avg_factor
    return loss

class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _transpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


if __name__ == "__main__":
    pred = torch.tensor([[125, 456, 321, 647],
                          [25, 321, 216, 645],
                          [111, 195, 341, 679],
                          [30, 134, 105, 371]],dtype=torch.float32)
    gt = torch.tensor([[132, 407, 301, 667],
                           [29, 322, 234, 664],
                           [109, 201, 315, 680],
                           [41, 140, 115, 384]],dtype=torch.float32)

    loss = RegGiouLoss
    giou_loss = _generalized_iou(pred+1000,gt)
    print(giou_loss)
