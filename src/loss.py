import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, cross_entropy, nll_loss, binary_cross_entropy_with_logits

from .utils import idcg
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def mask_nll_loss(inp, target, mask):
  # Calculate our loss based on our decoder’s output tensor, the target tensor,
  # and a binary mask tensor describing the padding of the target tensor.
  cross_entropy = -torch.log(torch.gather(inp, 2, target.unsqueeze(2)).squeeze(2))
  loss = cross_entropy.masked_select(mask).mean()

  # loss = loss.to(DEVICE)

  return loss

def mask_ce_loss(inp, target, mask):
  '''
  Calculate the cross entropy loss and a binary mask tensor describing the padding of the target tensor.

  Inputs:
    inp: tensor of raw values; size=(seq, batch, classes)
    target: tensor of classes; size=(seq, batch)
    mask: binary mask; size=(seq, batch)

  Outputs:
    loss: scalar
  '''

  # format the sizes
  inp = inp.view(-1, inp.size(-1))
  target = target.view(-1)

  # temp convert mask to byte coz pytorch 1.1.0
  mask = mask.view(-1).byte()

  loss = cross_entropy(inp, target, reduction='none').masked_select(mask).mean()

  return loss

def rmse_loss(preds, truth):
  return mse_loss(preds, truth).sqrt()

def bpr_loss(preds, truth, weight=None):
  assert preds.size(-1) == 2
  assert truth.size(-1) == 2

  diff = preds[..., 0] - preds[..., 1]
  target = truth[..., 0] - truth[..., 1]
  target[target > 0] = 1
  target[target == 0] = .5
  target[target < 0] = 0

  return binary_cross_entropy_with_logits(diff, target, weight=weight)

def lambda_rank_loss(preds, scores):
  '''
  (batch, item_size)
  '''

  batch_size = preds.size(0)
  item_size = preds.size(-1)

  _, indices = preds.sort(descending=True)

  # (batch)
  idcg_var = idcg(scores)

  ranks = torch.zeros(scores.size(), device=DEVICE)
  for i in range(batch_size):
    ranks[i][indices[i]] = torch.arange(1, item_size + 1, dtype=torch.float, device=DEVICE)

  pairs = []
  score_pairs = []
  weight = []

  # only use first item to form pairs
  # for i in range(item_size):
  for i in range(1):
    for j in range(i + 1, item_size):
      delta_ndcg = (2 ** scores[:, i] - 2 ** scores[:, j]) * \
        (
          1 / (ranks[:, i] + 1).log() - \
          1 / (ranks[:, j] + 1).log() \
        ) / idcg_var

      delta_ndcg = delta_ndcg.abs()
      weight.append(delta_ndcg)

      pairs.append(preds[:,(i,j)])
      score_pairs.append(scores[:,(i,j)])

  pairs = torch.cat(pairs)
  score_pairs = torch.cat(score_pairs)
  weight = torch.cat(weight)

  return bpr_loss(pairs, score_pairs, weight)


class RLRankingLoss(nn.Module):
  '''
  Inputs:
    actions: (seq, batch * grp_size)
    probs: (seq, batch * grp_size) policy probabilities of sampled actions
    hiddens: [seq (batch * grp_size)] latent state of each action to further generation
    review_lens: (batch * grp_size)
    scores: (batch, grp_size)

  Outputs:
    loss
  '''

  def __init__(self, ranker, generator, voc, loss_type, rollout_num=8, max_length=0):
    super().__init__()

    self.ranker = ranker
    self.rollout_num = rollout_num
    self.max_length = max_length

    self.voc = voc
    self.generator = generator

    self.loss_type = loss_type


  def _forward_bpr(self, actions, probs, hiddens, action_lens, scores):
    max_given_length = actions.size(0)

    assert self.max_length >= max_given_length

    target = scores[:, 0] - scores[:, 1]
    target[target > 0] = 1
    target[target == 0] = .5
    target[target < 0] = 0
    # expand to calculate rewards
    target = target.view(-1, 1).expand(-1, 2)

    base_ranks, _ = self.ranker(actions, rvw_lens=action_lens)

    # (batch * grp_size) -> (batch, grp_size)
    base_ranks = base_ranks.view(scores.size())

    rewards = torch.zeros(probs.size(), device=probs.device)

    # Monte Carlo Search with Rollout policy
    for i in range(self.rollout_num):
      # action 1 ... t-1
      for action_idx in range(max_given_length):
        rollout_ranks = base_ranks.detach()

        # nothing need to rollout when in the end
        if action_idx != max_given_length - 1:
          # only rollout for review whose length longer than current action
          need_rollout = action_lens > action_idx + 1

          # (layers * dir, batch, hidden)
          init_hidden = hiddens[action_idx][:, need_rollout]
          action_var = actions[action_idx][need_rollout]
          max_gen_len = self.max_length - action_idx - 1


          sampled_words, _, _, sampled_action_lens = self.generator.sample(init_hidden, action_var, max_length=max_gen_len, eos_idx=self.voc.eos_idx)

          # combine with given state & action
          sampled_words = torch.cat([
            actions[:action_idx + 1, need_rollout], sampled_words
          ], dim=0)
          sampled_action_lens = sampled_action_lens + action_idx + 1

          sampled_ranks, _ = self.ranker(sampled_words, rvw_lens=sampled_action_lens)

          rollout_ranks = rollout_ranks.view(-1)
          rollout_ranks[need_rollout] = sampled_ranks
          rollout_ranks = rollout_ranks.view(scores.size())

        # temp bpr like reward

        # diff of probs of 1st is larger than 2nd
        prob_diff = target - torch.stack([
          (rollout_ranks[:, 0] - base_ranks[:, 1]).sigmoid(),
          (base_ranks[:, 0] - rollout_ranks[:, 1]).sigmoid()
        ], dim=1)

        # TODO: only update for need_rollout
        rewards[action_idx] += 1 - prob_diff.view(-1).abs().detach()

    rewards /= self.rollout_num

    # temp use byte mask in torch 1.1.0
    action_mask = torch.zeros(actions.size(), dtype=torch.uint8, device=actions.device)

    for i, l in enumerate(action_lens):
      action_mask[:l, i] = 1

    # negative to minimize, the loss is meaningless
    return - (probs.log() * rewards).masked_select(action_mask).sum()


  def _forward_mse(self, actions, probs, hiddens, action_lens, scores, features=None):
    max_given_length = actions.size(0)

    assert self.max_length >= max_given_length

    scores = scores.view(-1)

    base_ranks, _ = self.ranker(actions, rvw_lens=action_lens)

    risks = torch.zeros(probs.size(), device=probs.device)

    # Monte Carlo Search with Rollout policy
    for i in range(self.rollout_num):
      # action 1 ... t-1
      for action_idx in range(max_given_length):
        rollout_ranks = base_ranks.detach()

        # nothing need to rollout when in the end
        if action_idx != max_given_length - 1:
          # only rollout for review whose length longer than current action
          need_rollout = action_lens > action_idx + 1

          # (layers * dir, batch, hidden)
          init_hidden = hiddens[action_idx][:, need_rollout]
          action_var = actions[action_idx][need_rollout]
          max_gen_len = self.max_length - action_idx - 1


          sampled_words, _, _, sampled_action_lens = self.generator.sample(init_hidden, action_var, max_length=max_gen_len, eos_idx=self.voc.eos_idx)

          # combine with given state & action
          sampled_words = torch.cat([
            actions[:action_idx + 1, need_rollout], sampled_words
          ], dim=0)
          sampled_action_lens = sampled_action_lens + action_idx + 1

          sampled_ranks, _ = self.ranker(sampled_words, rvw_lens=sampled_action_lens)

          rollout_ranks[need_rollout] = sampled_ranks

        # TODO: only update for need_rollout
        risks[action_idx] += mse_loss(rollout_ranks, scores, reduction='none').detach()

    risks /= self.rollout_num

    # (seq, batch, voc_size)
    features = features.unsqueeze(0).expand(actions.size(0), -1, -1)
    is_features = features.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    risks[is_features] *= 0.6

    # temp use byte mask in torch 1.1.0
    action_mask = torch.zeros(actions.size(), dtype=torch.uint8, device=actions.device)

    for i, l in enumerate(action_lens):
      action_mask[:l, i] = 1

    # negative to minimize, the loss is meaningless
    return (- probs.log() / risks).masked_select(action_mask).sum()

  def forward(self, *args):
    if self.loss_type == 'BPR':
      return self._forward_bpr(*args)
    elif self.loss_type == 'MSE':
      return self._forward_mse(*args)

