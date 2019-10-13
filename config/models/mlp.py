from .base import *

MODEL_TYPE = 'MultiTaskMLP'

# [Rater]

# input size depend on embedding size, output size is 1
SHARED_LAYER_SIZES = [128, 64, 32]
TASK_LAYER_SIZES = [16, 1]


# [Training]

BATCH_SIZE = 256
# the coefficient of the recommendation loss
LOSS_TYPE = 'BPR'
LR = 5e-4
L2_PENALTY = 1e-6

MATCH_TENSOR_TYPE = 'concat'
