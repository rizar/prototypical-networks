import os
import argparse
import json
import math
import numpy
from tqdm import tqdm

import torch
import torchnet as tnt

from protonets.utils import filter_opt, merge_dict
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
from protonets.data.omniglot import load_class_images

def main(opt):
    # load model
    model = torch.load(opt['model.model_path'])
    model.eval()

    classes = set()
    for root, dirs, files in os.walk('/home/dzmitry/Dist/prototypical-networks/data'):
      # a heuristic way to identify the directories with characters
      if files and not dirs and all(f.endswith('.png') for f in files):
        classes.add(os.path.join(*root.split('/')[-2:]))
    data = [load_class_images({'class': class_ + '/rot00'})['data'] for class_ in classes]
    data = torch.cat(data, 1).permute(1, 0, 2, 3).unsqueeze(2)

    n_ways = 20
    n_support = 10

    accs = []
    for i in tqdm(range(100)):
      class_ids = numpy.random.choice(list(range(data.shape[0])), n_ways)
      xs = data[class_ids, :n_support].contiguous()
      xq = data[class_ids, n_support:]
      xq = xq.view(xq.size(0) * xq.size(1), *xq.size()[2:])
      z_proto = model.compute_prototypes(xs)
      y_hat = model.predict_class(z_proto, xq)

      target_inds = torch.arange(0, n_ways).unsqueeze(1).expand(n_ways, 20 - n_support).long()
      target_inds = target_inds.contiguous().view(-1)
      acc = torch.eq(y_hat, target_inds).float().mean()
      accs.append(acc)
    print(numpy.mean(accs))


parser = argparse.ArgumentParser(description='Evaluate few-shot prototypical networks')

default_model_path = 'results/best_model.pt'
parser.add_argument('--model.model_path', type=str, default=default_model_path, metavar='MODELPATH',
                    help="location of pretrained model to evaluate (default: {:s})".format(default_model_path))
args = vars(parser.parse_args())

main(args)
