import json
import os
import numpy as np
join = os.path.join

orig_train_dir = "/lfs/local/local/prabhat8/transfer_learning/original_train_images1"
fake_train_dir = "/lfs/local/local/prabhat8/transfer_learning/fake_train_images1"
orig_test_dir = "/lfs/local/local/prabhat8/transfer_learning/original_test_images1"
fake_test_dir = "/lfs/local/local/prabhat8/transfer_learning/fake_test_images1"

orig_train_fnames = [join(orig_train_dir, fname) for fname in os.listdir(orig_train_dir)]
fake_train_fnames = [join(fake_train_dir, fname) for fname in os.listdir(fake_train_dir)]

orig_test_fnames = [join(orig_test_dir, fname) for fname in os.listdir(orig_test_dir)]
fake_test_fnames = [join(fake_test_dir, fname) for fname in os.listdir(fake_test_dir)]

print(len(orig_train_fnames))
print(len(orig_test_fnames))
print(len(fake_train_fnames))
print(len(fake_test_fnames))


def write_json(orig_fnames, fake_fnames, out_file) :
    fnames = orig_fnames + fake_fnames
    labels = [0]*len(orig_fnames) + [1]*len(fake_fnames)
    
    indices = np.arange(len(fnames))
    np.random.shuffle(indices)
    fnames = [fnames[idx] for idx in indices]
    labels = [labels[idx] for idx in indices]
    print(len(fnames))
    json_data = {}
    json_data['names'] = fnames
    json_data['labels'] = labels

    with open(out_file, 'w') as f :
        json.dump(json_data, f)

indices = np.arange(len(fake_train_fnames))
np.random.shuffle(indices)
fake_train_fnames = [fake_train_fnames[idx] for idx in indices]


fake_train_fnames = fake_train_fnames[:len(orig_train_fnames)]

indices = np.arange(len(fake_test_fnames))
np.random.shuffle(indices)
fake_test_fnames = [fake_test_fnames[idx] for idx in indices]


fake_test_fnames = fake_test_fnames[:len(orig_test_fnames)]



print(len(orig_train_fnames), len(fake_train_fnames), len(orig_test_fnames), len(fake_test_fnames))

print(orig_test_fnames[0:2])
print(fake_test_fnames[0:2])
write_json(orig_train_fnames, fake_train_fnames, "train1.json")
write_json(orig_test_fnames, fake_test_fnames, "test1.json")


