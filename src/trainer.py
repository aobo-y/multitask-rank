import math
import random
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss, cross_entropy, binary_cross_entropy_with_logits
from torch import optim

from .loss import mask_ce_loss, bpr_loss, lambda_rank_loss
from .data import ReviewGroupDataLoader, basic_builder
# from .evaluate import test_rate_ndcg, test_review_ndcg

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

def iter_2_device(iterable):
  ''' move tuple of data to device if tensor '''
  return (
    i.to(DEVICE) if torch.is_tensor(i) else i
    for i in iterable
  )

class AbstractTrainer:
  ''' Abstract Trainer Pipeline '''

  def __init__(self,
    model,
    ckpt_mng,
    batch_size=64,
    lr=.01,
    l2=0,
    clip=1.,
    patience=5,
    max_iters=None,
    save_every=5,
    grp_config=None
  ):
    self.model = model

    self.ckpt_mng = ckpt_mng

    self.batch_size = batch_size
    self.optimizer = optim.Adam(
      model.parameters(),
      lr=lr,
      weight_decay=l2
    )
    self.clip = clip

    # trained epochs
    self.trained_epoch = 0
    self.train_results = []
    self.val_results = []

    self.collate_fn = basic_builder

    self.patience = patience
    self.max_iters = float('inf') if max_iters is None else max_iters
    self.save_every = save_every

    self.ckpt_name = lambda epoch: str(epoch)

    self.grp_config = grp_config

  def log(self, *args):
    '''formatted log output for training'''

    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{time}   ', *args)

  def resume(self, checkpoint):
    '''load checkpoint'''

    self.trained_epoch = checkpoint['epoch']
    self.train_results = checkpoint['train_results']
    self.val_results = checkpoint['val_results']
    self.optimizer.load_state_dict(checkpoint['opt'])

  def reset_epoch(self):
    self.trained_epoch = 0
    self.train_results = []
    self.val_results = []

  def run_batch(self, training_batch, val=False):
    '''
    Run a batch of any batch size with the model

    Inputs:
      training_batch: train data batch created by batch_2_seq
      val: if it is for validation, no backward & optim
    Outputs:
      result: tuple (loss, *other_stats) of numbers or element tensor
        loss: a loss tensor to optimize
        other_stats: any other values to accumulate
    '''

    pass

  def train(self, train_data, dev_data):
    patience = self.patience # end the function when reaching threshold
    best_epoch = self._best_epoch()

    epoch = self.trained_epoch + 1

    # Data loaders with custom batch builder
    trainloader = ReviewGroupDataLoader(train_data, collate_fn=self.collate_fn, grp_config=self.grp_config, batch_size=self.batch_size, shuffle=True, num_workers=4)

    # maximum iteration per epoch
    iter_len = min(self.max_iters, len(trainloader))
    # culculate print every to ensure ard 5 logs per epoch
    PRINT_EVERY = 10 ** round(math.log10(iter_len / 5))

    self.log(f'Start training from epoch {epoch}...')


    while True:
      self.model.train()
      results_sum = []

      for idx, training_batch in enumerate(trainloader):
        if idx >= iter_len:
          break

        # run a training iteration with batch
        batch_result = self.run_batch(training_batch)
        if type(batch_result) != tuple:
          batch_result = (batch_result,)

        loss = batch_result[0]

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients: gradients are modified in place
        # _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        # Adjust model weights
        self.optimizer.step()

        # Accumulate results
        self._accum_results(results_sum, batch_result)

        # Print progress
        iteration = idx + 1
        if iteration % PRINT_EVERY == 0:
          print_result = self._sum_to_result(results_sum, iteration)
          self.log('Epoch {}; Iter: {} {:.1f}%; {};'.format(epoch, iteration, iteration / iter_len * 100, self._result_to_str(print_result)))

      self.trained_epoch = epoch
      epoch_result = self._sum_to_result(results_sum, iteration)
      self.train_results.append(epoch_result)

      # validation
      self.model.eval()
      val_result = self.validate(dev_data)
      self.model.train()

      self.log('Validation; Epoch {}; {};'.format(epoch, self._result_to_str(val_result)))

      self.val_results.append(val_result)

      # new best if no prev best or the sort key is smaller than prev best's
      is_new_best = best_epoch is None or \
         self._result_sort_key(val_result) < self._result_sort_key(self.val_results[best_epoch-1])

      self._handle_ckpt(epoch, is_new_best, best_epoch)

      # if better than before, recover patience; otherwise, lose patience
      if is_new_best:
        patience = self.patience
        best_epoch = epoch
      else:
        patience -= 1

      if not patience:
        break

      epoch += 1

    best_result = self.val_results[best_epoch-1]
    self.log('Training ends: best result {} at epoch {}'.format(self._result_to_str(best_result), best_epoch))


  def validate(self, dev_data):
    devloader = ReviewGroupDataLoader(dev_data, collate_fn=self.collate_fn, grp_config=self.grp_config, batch_size=self.batch_size, shuffle=False)

    results_sum = []

    for dev_batch in devloader:
      result = self.run_batch(dev_batch, val=True)
      if type(result) != tuple:
        result = (result,)

      # Accumulate results
      self._accum_results(results_sum, result)

    return self._sum_to_result(results_sum, len(devloader))

  def _result_to_str(self, epoch_result):
    ''' convert result list to readable string '''
    return 'Loss: {:.4f}'.format(epoch_result)

  def _sum_to_result(self, results_sum, length):
    '''
    Convert accumulated sum of results to epoch result
    by default return the average batch loss
    '''
    loss_sum = results_sum[0]
    return loss_sum / length

  def _accum_results(self, results_sum, batch_result):
    ''' accumulate batch result of run batch '''

    while len(results_sum) < len(batch_result):
      results_sum.append(0)
    for i, val in enumerate(batch_result):
      results_sum[i] += val.item() if torch.is_tensor(val) else val

  def _result_sort_key(self, result):
    ''' return the sorting value of a result, the smaller the better '''
    return result

  def _best_epoch(self):
    '''
    get the epoch of best result, smallest sort key value, from results savings when resumed from checkpoint
    '''

    best_val, best_epoch = math.inf, None

    for i, result in enumerate(self.val_results):
      val = self._result_sort_key(result)
      if val < best_val:
        best_val = val
        best_epoch = i + 1

    return best_epoch

  def _handle_ckpt(self, epoch, is_new_best, best_epoch):
    '''
    Always save a checkpoint for the latest epoch
    Remove the checkpoint for the previous epoch
    If the latest is the new best record, remove the previous best
    Regular saves are exempted from removes
    '''

    # save new checkpoint
    cp_name = self.ckpt_name(epoch)
    self.ckpt_mng.save(cp_name, {
      'epoch': epoch,
      'train_results': self.train_results,
      'val_results': self.val_results,
      'model': self.model.state_dict(),
      'opt': self.optimizer.state_dict()
    })
    self.log('Save checkpoint:', cp_name)

    epochs_to_purge = []
    # remove previous non-best checkpoint
    prev_epoch = epoch - 1
    if prev_epoch != best_epoch:
      epochs_to_purge.append(prev_epoch)

    # remove previous best checkpoint
    if is_new_best and best_epoch:
      epochs_to_purge.append(best_epoch)

    for e in epochs_to_purge:
      if e % self.save_every != 0:
        cp_name = self.ckpt_name(e)
        self.ckpt_mng.delete(cp_name)
        self.log('Delete checkpoint:', cp_name)


class RankerTrainer(AbstractTrainer):
  ''' Trainer to train recommendation model '''

  def __init__(self, *args, loss_type=None, **kargs):
    super().__init__(*args, **kargs)

    self.loss_type = loss_type
    self.loss_fn = {
      'BPR': bpr_loss,
      'LambdaRank': lambda_rank_loss,
      'MSE': mse_loss
    }[loss_type]

  def run_batch(self, training_batch, val=False):
    '''
    Outputs:
      loss: tensor, overall loss to optimize
    '''

    # extract fields from batch & set DEVICE options
    users, items, ratings = iter_2_device(training_batch)

    # flatten inputs
    rate_output = self.model(users.view(-1), items.view(-1))
    rate_output = rate_output.view(ratings.size()).transpose(1, 2).flatten(0, 1)

    ratings = ratings.transpose(1, 2).flatten(0, 1)

    # ignore both -1 cases
    indices = 1 - (ratings[:, 0] == -1) * (ratings[:, 1] == -1)

    rate_loss = self.loss_fn(rate_output[indices], ratings[indices])

    return rate_loss


  def validate(self, dev_data):
    if self.loss_type == 'LambdaRank':
      # loss is pointless in LambdaRank
      val_result = 0.
    else:
      val_result = super().validate(dev_data)

    return val_result

