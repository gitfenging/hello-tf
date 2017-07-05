# coding=utf8
import os
import sys
import gzip
import numpy as np
import PIL.Image as Image

SRC_DIR = 'mnist-download-data'
TRAIN_DATA_FILE = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS_FILE = 'train-labels-idx1-ubyte.gz'

OUTPUT_DIR = 'output/raw_image'

# 60000为图片个数，28×28为图片宽高
with gzip.open(os.path.join(SRC_DIR, TRAIN_DATA_FILE)) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(60000 * 28 * 28)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(60000, 28, 28)
    print('data OK')

with gzip.open(os.path.join(SRC_DIR, TRAIN_LABELS_FILE)) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(60000)
    labels = np.frombuffer(buf, dtype=np.int8).astype(np.int64)
    print('labels OK')

for idx, label in enumerate(labels):
    if not os.path.exists(os.path.join(OUTPUT_DIR, str(label))):
        os.makedirs(os.path.join(OUTPUT_DIR, str(label)), 0o700)
    img = Image.fromarray(data[idx], "L")
    img.save(os.path.join(OUTPUT_DIR, str(label), 'img' + str(idx) + '.jpg'))
    sys.stdout.write('\r处理中（%d / %d）' % (idx, 60000))
    sys.stdout.flush()
