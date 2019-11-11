"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from model import Model

import h5py
import numpy as np
import scipy.misc


def resize_images(data) :
  resized = []
  for x in data :
    resized.append(np.reshape(scipy.misc.imresize(x, (28, 28)), [-1]))
  return np.array(resized)

with h5py.File('usps.h5', 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]

X_te = X_te*255.0
num_eval_examples = len(X_te)
X_te = np.reshape(X_te, [num_eval_examples, 16, 16])
X_te = resize_images(X_te)
X_te = X_te/255.0


# Global constants
with open('config_usps.json') as config_file:
  config = json.load(config_file)

eval_batch_size = config['eval_batch_size']
eval_on_cpu = config['eval_on_cpu']

model_dir = config['model_dir']

if eval_on_cpu:
  with tf.device("/cpu:0"):
    model = Model()
    
else:
  model = Model()
  attack = LinfPGDAttack(model, 
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])

global_step = tf.contrib.framework.get_or_create_global_step()


# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
  os.makedirs(eval_dir)

last_checkpoint_filename = ''
already_seen_state = False

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(eval_dir)


# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename):
  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, filename)

    
    num_batches = int(len(X_te)/eval_batch_size)

    total_xent_nat = 0.
    total_corr_nat = 0

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = X_te[bstart:bend, :]
      y_batch = y_te[bstart:bend]

      dict_nat = {model.x_input: x_batch,
                  model.y_input: y_batch}

      cur_corr_nat, cur_xent_nat = sess.run(
                                      [model.num_correct,model.xent],
                                      feed_dict = dict_nat)

      total_xent_nat += cur_xent_nat
      total_corr_nat += cur_corr_nat

    avg_xent_nat = total_xent_nat / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples

    

    print('natural: {:.2f}%'.format(100 * acc_nat))
    print('avg nat loss: {:.4f}'.format(avg_xent_nat))

# Infinite eval loop
while True:
  cur_checkpoint = tf.train.latest_checkpoint(model_dir)
  print("the cur_checkpoint is ", cur_checkpoint)
  # Case 1: No checkpoint yet
  if cur_checkpoint is None:
    if not already_seen_state:
      print('No checkpoint yet, waiting ...', end='')
      already_seen_state = True
    else:
      print('.', end='')
    sys.stdout.flush()
    time.sleep(10)
  # Case 2: Previously unseen checkpoint
  elif cur_checkpoint != last_checkpoint_filename:
    print('\nCheckpoint {}, evaluating ...   ({})'.format(cur_checkpoint,
                                                          datetime.now()))
    sys.stdout.flush()
    last_checkpoint_filename = cur_checkpoint
    already_seen_state = False
    evaluate_checkpoint(cur_checkpoint)
  # Case 3: Previously evaluated checkpoint
  else:
    if not already_seen_state:
      print('Waiting for the next checkpoint ...   ({})   '.format(
            datetime.now()),
            end='')
      already_seen_state = True
    else:
      print('.', end='')
    sys.stdout.flush()
    time.sleep(10)
