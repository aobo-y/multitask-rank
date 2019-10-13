import os
import argparse

import config
from main import init
from src.trainer import RankerTrainer
from src.data import ReviewDataset

DIR_PATH = os.path.dirname(__file__)

KARGS_LOG_KEYS = {'batch_size', 'lr', 'l2', 'loss_type', 'patience', 'max_iters', 'grp_config'}

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', help='model name to save/load checkpoints')
  parser.add_argument('-c', '--checkpoint')
  args = parser.parse_args()

  model, misc = init(args.model, args.checkpoint)
  checkpoint, ckpt_mng, model_config = (
    misc[k] for k in ['checkpoint', 'ckpt_mng', 'model_config']
  )

  kargs = dict(
    batch_size=model_config.BATCH_SIZE,
    lr=model_config.LR,
    l2=model_config.L2_PENALTY,
    clip=model_config.CLIP,
    patience=config.PATIENCE,
    max_iters=model_config.MAX_ITERS,
    save_every=config.SAVE_EVERY,
    loss_type=model_config.LOSS_TYPE,
    grp_config=config.LOSS_TYPE_GRP_CONFIG[model_config.LOSS_TYPE]
  )

  print(f'Training config:', {k:v for k, v in kargs.items() if k in KARGS_LOG_KEYS})

  trainer = RankerTrainer(
    model,
    ckpt_mng,
    **kargs
  )

  if checkpoint:
      trainer.resume(checkpoint)
  else:
      ckpt_mng.save_meta()

  trainfile = os.path.join(DIR_PATH, config.TRAIN_CORPUS)
  devfile = os.path.join(DIR_PATH, config.DEV_CORPUS)

  print('Reading training data from %s...' % trainfile)

  train_datasets = ReviewDataset(trainfile)

  print(f'Read {len(train_datasets)} training reviews')

  print("Reading development data from %s..." % devfile)

  dev_datasets = ReviewDataset(devfile)

  print(f'Read {len(dev_datasets)} development reviews')

  # Ensure dropout layers are in train mode
  model.train()

  trainer.train(train_datasets, dev_datasets)

if __name__ == '__main__':
    main()
