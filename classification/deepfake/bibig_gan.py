import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image
import tensorflow.contrib.slim as slim
import os
import json
import cv2

def read_image(fname) :
	img = np.array(Image.open(fname))
	img_shape = list(img.shape[:2])
	min_side = min(img_shape)
	start = [int((img_shape[0] - min_side)//2), int((img_shape[1] - min_side)//2)]
	stop = [start[0]+min_side, start[1]+min_side]
	cropped_image = img[start[0]:stop[0], start[1]:stop[1]]
	resized_image = cv2.resize(img, (256, 256))
	return (resized_image/255.0)*2.0 - 1.0


images = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
labels = tf.placeholder(shape = [None], dtype = tf.int32)

module = hub.Module('https://tfhub.dev/deepmind/bigbigan-resnet50/1')
features = module(images, signature='encode', as_dict=True)

main_feature = features['avepool_feat']

out = slim.fully_connected(main_feature, 2, scope = "final_linear")
correct = tf.reduce_sum(tf.cast(tf.equal(labels, tf.cast(tf.argmax(out, axis = -1), dtype = tf.int32)), dtype = tf.float32))

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = out, labels = labels))
train_vars = tf.trainable_variables()
train_step = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(loss)

base_file = "train.json"
val_file = "test.json"

checkpoint_dir = "models"
if not os.path.exists(checkpoint_dir) :
	os.makedirs(checkpoint_dir)


with open(base_file, 'r') as f :	
	train_data = json.load(f)
	train_fnames = train_data['names']
	train_labels = train_data['labels']

with open(val_file, 'r') as f :
	val_data = json.load(f)
	val_fnames = val_data['names']
	val_labels = val_data['labels']

def train_loop(epoch, batch_size = 32) :
	print_freq = 10
	indices = np.arange(len(train_fnames))
	np.random.shuffle(indices)
	train_fnames_shuf = [train_fnames[idx] for idx in indices]
	train_labels_shuf = [train_labels[idx] for idx in indices]
	avg_loss = 0.0

	iters = int(len(train_fnames)/batch_size)
	print("the total train iters are ", iters)
	for i in range(iters) :
		fnames = train_fnames_shuf[i*batch_size:(i+1)*batch_size]
		inp_labels = np.array(train_labels_shuf[i*batch_size:(i+1)*batch_size]).astype(np.uint8)
		inp_images = np.array([read_image(fname) for fname in fnames])
		l, _ = sess.run([loss, train_step], feed_dict = {images : inp_images, labels : inp_labels})
		avg_loss = avg_loss + l
		if i % print_freq==0:
			print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, iters, avg_loss/float(i+1)  ))

def validate_loop(epoch, batch_size = 32) :
	corr = 0.0
	total = 0
	iters = int(len(val_fnames)/batch_size)
	print("the total val iters are ", iters)

	for i in range(iters) :
		fnames = val_fnames[i*batch_size:(i+1)*batch_size]
		inp_labels = np.array(val_labels[i*batch_size:(i+1)*batch_size]).astype(np.uint8)
		inp_images = np.array([read_image(fname) for fname in fnames])
		corr += sess.run(correct, feed_dict = {images : inp_images, labels : inp_labels})
		total += len(inp_images)

	acc = 100.0 * float(corr)/float(total)
	print('Epoch {:d} | Acc {:f}'.format(epoch, acc))
	return acc

def save_checkpoint() :
	saver.save(sess, os.path.join(checkpoint_dir, 'bibig_gan'))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

train_batch_size = 32
val_batch_size = 32

start_epoch = 0
end_epoch = 10

best_acc = -1.0

acc = validate_loop(0, val_batch_size)
for epoch in range(start_epoch, end_epoch) :

	train_loop(epoch, train_batch_size)
	acc = validate_loop(epoch, val_batch_size)
	if acc > best_acc :
		best_acc = acc
		save_checkpoint()









