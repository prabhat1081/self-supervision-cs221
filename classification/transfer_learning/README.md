### Data preparation

1. Download video data from [here](https://github.com/ondyari/FaceForensics/)

2. To sample images from video, perform face detection and split data into hard training and validation splits, run the following - 
```
python get_face_data_new1.py
python get_face_data_fake_new1.py
```
Otherwise to split into easy subset - 

```
python get_face_data.py
python get_face_data_fake.py
```
Make json files from the data by the following - 

```
python make_data_new.py

```

### Main scripts

1. ResNet50 (random initialization) - 

```
python train_resnet_random.py

```

2. ResNet50 (ImageNet pre-trained) - 

```
python train_resnet.py 

```
3. AlexNet (ImageNet pre-trained) - 

```
python train_orig.py
```

4. AlexNet (backbone trained with ImageNet and rotation)

```
python train_rotation.py
```

5. BigBiGAN (ResNet50 Encoder)

```
python bibig_gan.py
```

6. BigBiGAN (RevNet-50X4 Encoder)

```
python bibig_gan_large.py
```


