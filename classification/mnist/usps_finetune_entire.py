"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

from model import Model

import h5py
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

X_tr = X_tr*255.0
num_train_examples = len(X_tr)
X_tr = np.reshape(X_tr, [num_train_examples, 16, 16])
X_tr = resize_images(X_tr)
X_tr = X_tr/255.0


with open('config_usps.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

global_step = tf.contrib.framework.get_or_create_global_step()
model = Model()

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent,
                                                   global_step=global_step)


# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)


shutil.copy('config.json', model_dir)

indices = np.arange(num_train_examples)
epochs = 5

with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  sess.run(tf.global_variables_initializer())

  # saver.restore(sess, "models/natural/checkpoint-24900")
  saver.restore(sess, "models/natural_rotation/checkpoint-15000")
  training_time = 0.0

  # Main training loop
  for i in range(epochs) :
    np.random.shuffle(indices)
    train_iters = int(num_train_examples/batch_size)
    for ii in range(train_iters):
      bstart = ii * batch_size
      bend = min(bstart + batch_size, num_train_examples)

      x_batch = X_tr[bstart:bend, :]
      y_batch = y_tr[bstart:bend]
      # x_batch, y_batch = mnist.train.next_batch(batch_size)

      # Compute Adversarial Perturbations
      start = timer()
      end = timer()
      training_time += end - start

      nat_dict = {model.x_input: x_batch,
                  model.y_input: y_batch}

      # Output to stdout
      if ii % num_output_steps == 0:
        nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
        print('Step {}:    ({})'.format(ii, datetime.now()))
        print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
        if ii != 0:
          print('    {} examples per second'.format(
              num_output_steps * batch_size / training_time))
          training_time = 0.0
      

      # Write a checkpoint
      
        

      # Actual training step
      start = timer()
      sess.run(train_step, feed_dict=nat_dict)
      end = timer()
      training_time += end - start

    saver.save(sess,
                   os.path.join(model_dir, 'checkpoint'),
                   global_step=global_step)

