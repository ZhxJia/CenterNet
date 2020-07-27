from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.smk3d import SmkDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from models.losses import RegGiouLoss
dataset_factory = {
    'coco': COCO,
    'pascal': PascalVOC,
    'kitti': KITTI,
    'coco_hp': COCOHP
}

_sample_factory = {
    'exdet': EXDetDataset,
    'ctdet': CTDetDataset,
    'ddd': DddDataset,
    'multi_pose': MultiPoseDataset,
    'smk': SmkDataset
}


# 同时继承数据集和数据集获取类
def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass

    return Dataset


if __name__ == "__main__":
    from opts import opts
    import torch
    loss = RegGiouLoss()
    dataset = get_dataset('kitti', 'smk')
    opt = opts().parse()
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)
    ds = dataset(opt, 'val')
    dl = torch.utils.data.DataLoader(ds, batch_size=1)
    batch = next(iter(dl))
    gt_boxes = batch['wh']
    weight = batch['reg_weight']
    target_hm = batch['hm']
    pred_hm = target_hm
    # # test giou loss
    # torch.random.manual_seed(123)
    # loc_x = torch.arange(0, 128, dtype=torch.float32)
    # loc_y = torch.arange(0, 128, dtype=torch.float32)
    # loc_y, loc_x = torch.meshgrid(loc_y, loc_x)
    # base_loc = torch.stack((loc_y, loc_x), dim=0)
    # predict_wh = torch.cat((base_loc - gt_boxes[:, [0,1]],
    #                         -base_loc + gt_boxes[:,[2,3]]),dim=1)  + torch.rand(1,4,128,128,dtype=torch.float32)*20
    # reg_loss = loss(pred_hm,predict_wh,target_hm,gt_boxes,weight)
    print(batch)
    # print(reg_loss)


