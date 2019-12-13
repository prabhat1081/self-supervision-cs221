import numpy as np
import glob
import dlib
import cv2
from pathlib import Path
import glob
import os
from PIL import Image
import multiprocessing

join = os.path.join

face_detector = dlib.get_frontal_face_detector()
frames_per_video = 115

def make_dir(dir_name) :
        if not os.path.exists(dir_name) :
                os.makedirs(dir_name)

orig_dir = "original_images"
fake_dir = "fake_images"
fake_train_dir = "fake_train_images1"
fake_test_dir = "fake_test_images1"

make_dir(fake_train_dir)
make_dir(fake_test_dir)

orig_count = 0

def save_image(img, fname) :
	img = img[:,:,::-1]
	img = Image.fromarray(img.astype(np.uint8))
	img.save(fname)

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
	"""
	Expects a dlib face to generate a quadratic bounding box.
	:param face: dlib face class
	:param width: frame width
	:param height: frame height
	:param scale: bounding box size multiplier to get a bigger face region
	:param minsize: set minimum bounding box size
	:return: x, y, bounding_box_size in opencv form
	"""
	x1 = face.left()
	y1 = face.top()
	x2 = face.right()
	y2 = face.bottom()
	size_bb = int(max(x2 - x1, y2 - y1) * scale)
	if minsize:
		if size_bb < minsize:
			size_bb = minsize
	center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

	# Check for out of bounds, x-y top left corner
	x1 = max(int(center_x - size_bb // 2), 0)
	y1 = max(int(center_y - size_bb // 2), 0)
	# Check for too big bb size for given x, y
	size_bb = min(width - x1, size_bb)
	size_bb = min(height - y1, size_bb)

	return x1, y1, size_bb


def read_video(video_path) :
	reader = cv2.VideoCapture(video_path)
	frame_idx = 0
	num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
	sample_idx = int(num_frames/frames_per_video)
	actor_idx = int(video_path.split("/")[-1].split("_")[0])
	if sample_idx == 0 :
		return
	while reader.isOpened():
		_, image = reader.read()
		if image is None:
			break
		if frame_idx%sample_idx == 0 :

			height, width = image.shape[:2]
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			faces = face_detector(gray, 1)
			
			if len(faces):
				# For now only take biggest face
				face = faces[0]

				# --- Prediction ---------------------------------------------------
				# Face crop with dlib and bounding box scale enlargement
				x, y, size = get_boundingbox(face, width, height)
				cropped_face = image[y:y+size, x:x+size]
				fname = "_".join(video_path.split("/"))
				fname = fname.split(".")[0]+ str(frame_idx)
				fname = fname+".png"

				if not actor_idx in [25, 26, 27, 28] :
					save_image(cropped_face, join(fake_train_dir, fname))
				else :
					save_image(cropped_face, join(fake_test_dir, fname))

				

			else :
				continue
		frame_idx += 1



base_dir = "/lfs/local/local/prabhat8/deepfake/manipulated_sequences/"
files = Path(base_dir).rglob('*.mp4')
# for file in files :
# 	paths.append(file.absolute().as_posix())

orig_videos = [join(base_dir, file.absolute().as_posix()) for file in files]
indices = np.arange(len(orig_videos))
np.random.shuffle(indices)
indices = indices[:400]
orig_videos = [orig_videos[idx] for idx in indices]

with multiprocessing.Pool(20) as pool:
	pool.map(read_video, orig_videos)
	




		

