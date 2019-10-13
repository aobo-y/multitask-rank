import os
import json
import itertools
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader

from .utils.data import binary_mask

DIR_PATH = os.path.dirname(__file__)
ENV = os.environ['ENV'] if 'ENV' in os.environ else None

class Review:
  def __init__(self, user, item, ratings):
    self.user = user
    self.item = item
    self.ratings = ratings

class ReviewDataset(Dataset):
  def __init__(self, filepath):
    self.reviews = self.load(filepath)

    self.user_dict = defaultdict(dict)
    for review in self.reviews:
      self.user_dict[review.user][review.item] = review

  def load(self, filepath):
    def parse(line):
      entry = line.strip().split(':')
      user_idx = int(entry[1].split('\t')[0])
      item_idx = int(entry[2].split('\t')[0])

      # first 8 ratings
      ratings = entry[3].strip().split('\t')[:8]
      ratings = [float(r) for r in ratings]

      return Review(user_idx, item_idx, ratings)

    # Read the file and split into lines
    with open(filepath, encoding='utf-8') as f:
      lines = f.read().split('\n')

      # for fast development, cut 5000 samples
      if ENV == 'DEV':
        lines = lines[:5000]

    # Map every line into review
    return [parse(l) for l in lines if l != '']

  # Return review
  def __getitem__(self, idx):
      return self.reviews[idx]

  # Return the number of elements of the dataset.
  def __len__(self):
      return len(self.reviews)

  def get_review(self, uid, iid):
    return self.user_dict[uid][iid]

  @property
  def item_ids(self):
    return set(r.item for r in self.reviews)

  @property
  def user_ids(self):
    return set(r.user for r in self.reviews)

class ReviewGroupDataset(Dataset):
  '''
  Wrap ReviewDataset to group by user
    review_dataset: ReviewDataset
    grp_config:
      grp_size
      n_min_rated
  '''
  def __init__(self, review_dataset, grp_config=None):
    self.review_dataset = review_dataset
    self.user_dict = review_dataset.user_dict

    self.grp_size = grp_config['grp_size']
    self.n_min_rated = grp_config['n_min_rated']

    # rm user with only one item for bpr
    single_item_users = set([
      u for u, i in self.review_dataset.user_dict.items()
      if len(i) == 1
    ])

    self.review_dataset.reviews = [r for r in self.review_dataset.reviews if r.user not in single_item_users]

    assert self.n_min_rated <= self.grp_size

    self.max_item_id = max(r.item for r in self.review_dataset)

  def __getitem__(self, idx):
    rvw = self.review_dataset[idx]
    samples = [rvw]

    if self.n_min_rated > 1:
      # remove itself
      pool = [r for r in self.user_dict[rvw.user].values() if r != rvw]

      if len(pool) < self.n_min_rated - 1:
        raise Error('Not enough rated items in the user pool')
      else:
        samples += random.sample(pool, k=self.n_min_rated - 1)

    while len(samples) < self.grp_size:
      iid = random.randint(0, self.max_item_id)

      if iid in self.user_dict[rvw.user]:
        samples.append(self.user_dict[rvw.user][iid])
      else:
        samples.append(Review(rvw.user, iid, 0))

    return samples

  def __len__(self):
      return len(self.review_dataset)

def basic_builder(samples):
  users = torch.tensor([s.user for s in samples])
  items = torch.tensor([s.item for s in samples])
  ratings = torch.tensor([s.ratings for s in samples])

  return users, items, ratings

def grp_wrap_collate(collate_fn):
  def wrapped(grp_samples):
    '''
    Inputs:
      grp_samples: [batch, grp_size] nested array
    Outputs:
      users: (batch, grp_size)
      items: (batch, grp_size)
      scores: (batch, grp_size, rating_size)

      mask: (word_seq, batch)
    '''

    batch_size = len(grp_samples)
    grp_size = len(grp_samples[0])

    # flatten groups
    samples = sum(grp_samples, [])

    batch_data = collate_fn(samples)

    # hardcode i < 2 for users, items
    batch_data = [
      t.view(batch_size, -1) if i < 2 else t for i, t in enumerate(batch_data)
    ]
    batch_data[2] = batch_data[2].view(batch_size, grp_size, -1)

    return batch_data

  return wrapped

class ReviewGroupDataLoader(DataLoader):
  '''
  Review Dataset Loader supporting group by user
    dataset: ReviewDataset
    collate_fn: function
    grp_config:
      grp_size: number of reviews per group, default 0 for no group
      n_min_rated: minimum number of rated items, default None for all rated
  '''

  def __init__(self, dataset, collate_fn=None, grp_config=None, **kargs):
    if grp_config is not None:
      # need group reviews
      dataset = ReviewGroupDataset(dataset, grp_config=grp_config)

      collate_fn = grp_wrap_collate(collate_fn)


    super().__init__(dataset, collate_fn=collate_fn, **kargs)
