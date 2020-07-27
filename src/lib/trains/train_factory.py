from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer
from .smk3d import Smk3dTrainer

train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'smk3d': Smk3dTrainer,
  'ctdet': CtdetTrainer,
  'multi_pose': MultiPoseTrainer, 
}
