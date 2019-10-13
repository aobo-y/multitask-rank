import random
import torch
from torch import nn

class MultiTaskMLP(nn.Module):
  '''
  Multilayer Perceptron

  Inputs:
    input_var: shape=(batch_size, user_ebd_size + item_ebd_size)

  Outputs:
    output_var: shape=(batch_size)
    layer_outputs: list of tensors of shape=(batch_size, X_layer_size)
  '''

  def __init__(self,
    n_users,
    n_items,
    ebd_size,

    n_tasks,

    shared_layer_sizes,
    task_layer_sizes
  ):
    super().__init__()

    assert task_layer_sizes[-1] == 1

    self.user_ebd = nn.Embedding(n_users, ebd_size)
    self.item_ebd = nn.Embedding(n_items, ebd_size)

    shared_layers = []
    i_size = 2 * ebd_size
    for o_size in shared_layer_sizes:
      shared_layers.append(nn.Linear(i_size, o_size))
      shared_layers.append(nn.ReLU())
      i_size = o_size

    self.shared_layers = nn.Sequential(*shared_layers)

    task_layers_list = []
    for i in range(n_tasks):
      task_layers = []
      t_i_size = i_size

      for t_o_size in task_layer_sizes:
        task_layers.append(nn.Linear(t_i_size, t_o_size))
        task_layers.append(nn.ReLU())
        t_i_size = t_o_size

      # rm last activation
      task_layers = task_layers[:-1]

      task_layers_list.append(nn.Sequential(*task_layers))

    self.task_layers_list = nn.ModuleList(task_layers_list)


  def forward(self, user_var, item_var):
    user_vct = self.user_ebd(user_var)
    item_vct = self.item_ebd(item_var)

    input_var = torch.cat([user_vct, item_vct], dim=1)

    shared_var = self.shared_layers(input_var)
    task_var = [
      task_layers(shared_var)
      for task_layers in self.task_layers_list
    ]
    task_var = torch.cat(task_var, dim=1)

    return task_var
