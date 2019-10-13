import os
import argparse

import config
from main import init
from src.data import ReviewDataset
from src.evaluate import test_rate_mse, test_rate_ndcg, load_ndcg

DIR_PATH = os.path.dirname(__file__)

def load_test_dataset():
  testfile = os.path.join(DIR_PATH, config.TEST_CORPUS)
  print("Reading Testing data from %s..." % testfile)

  test_dataset = ReviewDataset(testfile)

  print(f'Read {len(test_dataset)} testing reviews')
  return test_dataset

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', help='model name to save/load checkpoints')
  parser.add_argument('-c', '--checkpoint')
  parser.add_argument('evals', nargs='+')
  args = parser.parse_args()

  model, misc = init(args.model, args.checkpoint)
  model.eval()

  test_dataset = load_test_dataset()

  for ev in args.evals:
    if ev == 'rmse':
      mse = test_rate_mse(test_dataset, model)
      print('Rate RMSE: ', mse)

    elif ev == 'ndcg':
      ndcg_path = os.path.join(DIR_PATH, 'data/ndcg_150.ls')
      ndcg_user_items = load_ndcg(ndcg_path)

      print('User size:', len(ndcg_user_items))

      vals = next(iter(ndcg_user_items.values()))
      size = len(vals)

      avg_ndcg, ndcg = test_rate_ndcg(model, test_dataset, ndcg_user_items)
      print(f'Rate NDCG({size}):', avg_ndcg, ndcg)

if __name__ == '__main__':
  main()
