import json
import os
import numpy as np
join = os.path.join

orig_dir = "/lfs/local/local/prabhat8/transfer_learning/original_images"
fake_dir = "/lfs/local/local/prabhat8/transfer_learning/fake_images"

orig_fnames = [join(orig_dir, fname) for fname in os.listdir(orig_dir)]
fake_fnames = [join(fake_dir, fname) for fname in os.listdir(fake_dir)]

print(len(orig_fnames))
print(len(fake_fnames))

orig_train_size = 33000
fake_train_size = 33000

def write_json(orig_fnames, fake_fnames, out_file) :

	fnames = orig_fnames + fake_fnames
	labels = [0]*len(orig_fnames) + [1]*len(fake_fnames)

	indices = np.arange(len(fnames))
	np.random.shuffle(indices)
	fnames = [fnames[idx] for idx in indices]
	labels = [labels[idx] for idx in indices]

	json_data = {}
	json_data['names'] = fnames
	json_data['labels'] = labels

	with open(out_file, 'w') as f :
		json.dump(json_data, f)

orig_train_fnames = orig_fnames[0:orig_train_size]
fake_train_fnames = fake_fnames[0:fake_train_size]

orig_test_fnames = orig_fnames[orig_train_size:37000]
fake_test_fnames = fake_fnames[fake_train_size:37000]

print(len(orig_train_fnames), len(fake_train_fnames), len(orig_test_fnames), len(fake_test_fnames))


write_json(orig_train_fnames, fake_train_fnames, "train.json")
write_json(orig_test_fnames, fake_test_fnames, "test.json")


