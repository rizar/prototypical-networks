import os
import sys
import argparse
import json
import math
import numpy
from tqdm import tqdm
from PIL import Image

import torch

from protonets.data.omniglot import load_class_images

def main(args):
    # load model
    model = torch.load(args.model_path)
    model.eval()

    classes = []
    for root, dirs, files in os.walk(args.data):
      # a heuristic way to identify the directories with characters
      if files and not dirs and all(f.endswith('.png') for f in files):
        class_ = os.path.join(*root.split('/')[-2:])
        classes.append(class_)
    print("Loaded data for {} classes".format(len(classes)))
    data = [load_class_images({'class': class_ + '/rot00'})['data']
            for class_ in classes]
    data = torch.cat(data, 1).permute(1, 0, 2, 3).unsqueeze(2)

    data = data[:, :args.examples_per_class].contiguous()
    z_proto = model.compute_prototypes(data)

    for path in sys.stdin.readlines():
      path = path.strip()
      image = Image.open(path)
      image = image.resize((28, 28))
      image = (1.0 - torch.from_numpy(numpy.array(image, numpy.float32, copy=False))
               .transpose(0, 1).contiguous().unsqueeze(0))
      xq = image.unsqueeze(0)
      y_hat = model.predict_class(z_proto, xq)
      print("{}: {}".format(path, classes[y_hat[0]]))


parser = argparse.ArgumentParser(description='Evaluate few-shot prototypical networks')

default_model_path = 'results/best_model.pt'
parser.add_argument('--model_path', type=str, default=default_model_path, metavar='MODELPATH',
                    help="location of pretrained model to evaluate (default: {:s})".format(default_model_path))
parser.add_argument('--data', type=str, required=True,
                    help="the location of the data")
parser.add_argument('--examples-per-class', type=int, default=10,
                    help="number of examples per class")
args = parser.parse_args()

main(args)
