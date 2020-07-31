from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, L1Loss, BinRotLoss
from models.decode import smk3d_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import smk3d_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer
from models.coder import SMOKECoder
from models.utils import _transpose_and_gather_feat, _gather_mask_feat


class Smk3dLoss(torch.nn.Module):
    def __init__(self, opt):
        super(Smk3dLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_reg = L1Loss() if opt.reg_loss == 'l1' else \
            L1Loss() if opt.reg_loss == 'disl1' else None
        self.opt = opt
        self.smoke_coder = SMOKECoder(opt.reference_depth, opt.reference_size, device='cuda')  # todo debug: device?

    def forward(self, outputs, batch):
        opt = self.opt

        hm_loss, ori_loss, dim_loss, loc_loss = 0, 0, 0, 0
        wh_loss, off_loss = 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            output['hm'] = _sigmoid(output['hm'])
            predict_box3d = self.prepare_prediction(output, batch)

            hm_loss = self.crit(output['hm'], batch['hm'])
            target_box3d = batch['reg']

            if self.opt.reg_loss == 'disl1':
                ori_loss = self.crit_reg(predict_box3d['ori'], batch['reg'], batch['reg_mask']) # todo debug： rot_mask or reg_mask
                dim_loss = self.crit_reg(predict_box3d['dim'], batch['reg'], batch['reg_mask'])
                loc_loss = self.crit_reg(predict_box3d['loc'], batch['reg'], batch['reg_mask'])
                reg_loss = opt.ori_weight * ori_loss + opt.dim_weight * dim_loss + opt.loc_weight * loc_loss
            elif self.opt.reg_loss == 'l1':
                reg_loss = self.crit_reg(predict_box3d, batch['reg'], batch['reg_mask']) \
                           * (opt.ori_weight + opt.dim_weight + opt.loc_weight) / 3

        loss = opt.hm_weight * hm_loss + reg_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'reg_loss': reg_loss,
                      'ori_loss': ori_loss, 'dim_loss': dim_loss, 'loc_loss': loc_loss}
        return loss, loss_stats

    def prepare_prediction(self, preds, targets):
        ind = targets['ind']
        mask = targets['reg_mask']
        pred_depths_offset = _transpose_and_gather_feat(preds['dep'], ind)  # Nx1x96x320 -> Nx50x1
        pred_proj_offsets = _transpose_and_gather_feat(preds['offset'], ind)  # Nx2x96x320 -> Nx50x2
        pred_dimensions_offsets = _transpose_and_gather_feat(preds['dim'], ind)  # Nx3x96x320 ->
        pred_orientation = _transpose_and_gather_feat(preds['rot'], ind)  # Nx2x96x320 ->

        pred_depths = self.smoke_coder.decode_depth(pred_depths_offset)  # Nx50x1

        pred_dimensions = self.smoke_coder.decode_dimension(targets['cls_ids'], pred_dimensions_offsets)

        pred_locations = self.smoke_coder.decode_location(targets['proj_cts'],
                                                          pred_proj_offsets,
                                                          pred_depths,
                                                          pred_dimensions,
                                                          targets['K'],
                                                          targets['trans_mat'])
        pred_rotys, _ = self.smoke_coder.decode_orientation(pred_orientation,
                                                            targets['locs'])

        if self.opt.reg_loss == 'disl1':
            pred_box3d_rotys = self.smoke_coder.encode_box3d(pred_rotys,
                                                             targets['dim'],
                                                             targets['locs'])
            pred_box3d_dims = self.smoke_coder.encode_box3d(targets['rotys'],
                                                            pred_dimensions,
                                                            targets['locs'])
            pred_box3d_locs = self.smoke_coder.encode_box3d(targets['rotys'],
                                                            targets['dim'],
                                                            pred_locations)
            return dict(ori=pred_box3d_rotys, dim=pred_box3d_dims, loc=pred_box3d_locs)

        elif self.opt.reg_loss == "l1":
            pred_box_3d = self.smoke_coder.encode_box3d(pred_rotys,
                                                        pred_dimensions,
                                                        pred_locations)
            return pred_box_3d


class Smk3dTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(Smk3dTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'reg_loss', 'ori_loss', 'dim_loss',
                       'loc_loss']
        loss = Smk3dLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        # wh = output['wh'] if opt.reg_bbox else None
        # reg = output['offset'] if opt.reg_offset else None
        dets = smk3d_decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], output['offset'], K=opt.K)

        # x, y, score, r1-r8, depth, dim1-dim3, cls
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        calib = batch['meta']['calib'].detach().numpy()
        # x, y, score, rot, depth, dim1, dim2, dim3
        # if opt.dataset == 'gta':
        #   dets[:, 12:15] /= 3
        dets_pred = smk3d_post_process(
            dets.copy(), batch['meta']['c'].detach().numpy(),
            batch['meta']['s'].detach().numpy(), calib, opt)
        dets_gt = smk3d_post_process(
            batch['meta']['gt_det'].detach().numpy().copy(),
            batch['meta']['c'].detach().numpy(),
            batch['meta']['s'].detach().numpy(), calib, opt)
        # for i in range(input.size(0)):
        for i in range(1):
            debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug == 3),
                                theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.opt.std + self.opt.mean) * 255.).astype(np.uint8)
            pred = debugger.gen_colormap(
                output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'hm_pred')
            debugger.add_blend_img(img, gt, 'hm_gt')
            # decode
            debugger.add_ct_detection(
                img, dets[i], show_box=opt.reg_bbox, center_thresh=opt.center_thresh,
                img_id='det_pred')
            debugger.add_ct_detection(
                img, batch['meta']['gt_det'][i].cpu().numpy().copy(),
                show_box=opt.reg_bbox, img_id='det_gt')
            debugger.add_3d_detection(
                batch['meta']['image_path'][i], dets_pred[i], calib[i],
                center_thresh=opt.center_thresh, img_id='add_pred')
            debugger.add_3d_detection(
                batch['meta']['image_path'][i], dets_gt[i], calib[i],
                center_thresh=opt.center_thresh, img_id='add_gt')
            # debugger.add_bird_view(
            #   dets_pred[i], center_thresh=opt.center_thresh, img_id='bird_pred')
            # debugger.add_bird_view(dets_gt[i], img_id='bird_gt')
            debugger.add_bird_views(
                dets_pred[i], dets_gt[i],
                center_thresh=opt.center_thresh, img_id='bird_pred_gt')

            # debugger.add_blend_img(img, pred, 'out', white=True)
            debugger.compose_vis_add(
                batch['meta']['image_path'][i], dets_pred[i], calib[i],
                opt.center_thresh, pred, 'bird_pred_gt', img_id='out')
            # debugger.add_img(img, img_id='out')
            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        opt = self.opt
        # wh = output['wh'] if opt.reg_bbox else None
        # reg = output['reg'] if opt.reg_offset else None
        dets = smk3d_decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], hm_offset=output['offset'], K=opt.K)

        # x, y, score, r1-r8, depth, dim1-dim3, cls
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        calib = batch['meta']['calib'].detach().numpy()
        # x, y, score, rot, depth, dim1, dim2, dim3
        dets_pred = smk3d_post_process(
            dets.copy(), batch['meta']['c'].detach().numpy(),
            batch['meta']['s'].detach().numpy(), calib, opt)
        img_id = batch['meta']['img_id'].detach().numpy()[0]
        results[img_id] = dets_pred[0]
        for j in range(1, opt.num_classes + 1):
            keep_inds = (results[img_id][j][:, -1] > opt.center_thresh)
            results[img_id][j] = results[img_id][j][keep_inds]
