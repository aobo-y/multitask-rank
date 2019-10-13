# Corpus, path relateds to /src
TRAIN_CORPUS = 'data/split/train.txt'
DEV_CORPUS = 'data/split/val.txt'
TEST_CORPUS = 'data/split/test.txt'

# Checkpoints relevant
SAVE_DIR = 'checkpoints'

# Trainer behaviors
PATIENCE = 5
PRINT_EVERY = 1000
SAVE_EVERY = 5


# Hardcode Embeddings size
N_USERS = 100005
N_ITEMS = 17257

N_TASKS = 8

LOSS_TYPE_GRP_CONFIG = {
  'MSE': None,
  'BPR': {'grp_size': 2, 'n_min_rated': 2},
  'LambdaRank': {'grp_size': 44, 'n_min_rated': 1},
}
