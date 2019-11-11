import h5py
import numpy as np
from PIL import Image
import scipy.misc

with h5py.File('usps.h5', 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]


def save_images(img, fname) :
	img = img*255.0
	img = np.reshape(img, [16, 16])
	img = scipy.misc.imresize(img, (28, 28))
	# img = img*255.0
	img = Image.fromarray(img.astype(np.uint8))
	img.save(fname)

print(X_tr.shape)
print(X_te.shape)

print(np.min(X_te), np.max(X_te))
# print(np.min(y_te), np.max(y_te))
print(y_te.shape)

save_images(X_te[0], "img0.png")

