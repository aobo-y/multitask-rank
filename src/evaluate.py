import os
import math
import random
import json
import pickle
from collections import Counter
from statistics import mean

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss, cross_entropy, binary_cross_entropy_with_logits

from .data import Review, ReviewDataset, grp_wrap_collate, basic_builder
from .loss import mask_ce_loss
from .utils import ndcg

DIR_PATH = os.path.dirname(__file__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_ndcg(ndcg_path):
  ndcg_user_items = pickle.load(open(ndcg_path, 'rb'))
  return ndcg_user_items

def test_rate_ndcg(model, test_data, ndcg_user_items=None, ndcg_path=None):
  if ndcg_user_items is None:
    with open(ndcg_path) as f:
      ndcg_user_items = load_ndcg(ndcg_path)

  ndcg_sum = [0] * 8
  batch_size = 64

  length = len(ndcg_user_items)
  user_items = list(ndcg_user_items.items())

  for i in range(0, len(user_items), batch_size):
    batch_user_items = user_items[i:i+batch_size]
    reviews = []

    for uid, items in batch_user_items:
      items = items[:150]

      reviews += [
        test_data.get_review(uid, iid) if iid in test_data.user_dict[uid] else Review(uid, iid, [0.] * 8)
        for iid in items
      ]


    batch = basic_builder(reviews)
    users, items, ratings = (i.to(DEVICE) for i in batch[:3])

    output = model(users, items).view(len(batch_user_items), -1, ratings.size(-1))

    ratings = ratings.view(len(batch_user_items), -1, ratings.size(-1))

    ratings[ratings == -1] = 0

    for i in range(ratings.size(-1)):
      _, indices = output[:, :, i].sort(descending=True)
      ranked_scores = ratings[:, :, i].gather(-1, indices)

      nn = ndcg(ranked_scores, k=10)
      print(nn)
      ndcg_sum[i] += nn.sum().item() #ndcg(ranked_scores, k=10).sum().item()

  ndcgs = [n / length for n in ndcg_sum]
  return mean(ndcgs), ndcgs


def test_rate_mse(test_data, model, voc):
  testloader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=basic_builder)

  total_loss = 0
  for batch in testloader:
    users, items, scores = (i.to(DEVICE) for i in batch[:3])

    rate_output = model.rate(users, items)
    total_loss += mse_loss(rate_output, scores, reduction='sum').item()

  rmse = math.sqrt(total_loss / len(test_data))

  return rmse
