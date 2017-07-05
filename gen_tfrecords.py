# coding=utf8
import os
import numpy as np
import PIL.Image as Image
import sys

import tensorflow as tf

import dataset_util

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'dataset_dir',
    'output/data-tfrecord',  # 要写绝对路径，否则TFRecordWriter会报错
    '存储mnist图片的tfrecord文件')
tf.app.flags.DEFINE_string(
    'image_data_dir',
    'output/raw_image',
    '图片原始数据存放位置')

_CLASS_NAME = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']


def _extract_images():
    labels = os.listdir(FLAGS.image_data_dir)
    result_data = []
    result_labels = []
    for i, label in enumerate(labels):
        sys.stdout.write('\r已读取: %d' % i)
        sys.stdout.flush()
        imgsname = os.listdir(os.path.join(FLAGS.image_data_dir, str(label)))
        for imgname in imgsname:
            img = Image.open(os.path.join(FLAGS.image_data_dir, str(label), imgname))
            data = np.asarray(img).astype(np.uint8)
            data = data.reshape([28, 28, 1])
            result_data.append(data)
            result_labels.append(np.int64(label))
    print('\n 数据读取完毕')
    return result_data, result_labels


images, labels = _extract_images()
if not tf.gfile.Exists(FLAGS.dataset_dir):
    tf.gfile.MakeDirs(FLAGS.dataset_dir)
tf_filepath = os.path.join(FLAGS.dataset_dir, 'mnist.tfrecord')
with tf.python_io.TFRecordWriter(tf_filepath) as tfrecord_writer:
    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=[28, 28, 1])
        encode_png = tf.image.encode_png(image)
        print('正在写入tfrecord...')
        with tf.Session() as sess:
            images_size = len(images)
            for i in range(images_size):
                png_string = sess.run(encode_png, {image: images[i]})
                example = dataset_util.image_to_tfexample(png_string, 'png'.encode(), 28, 28, labels[i])
                tfrecord_writer.write(example.SerializeToString())
                sys.stdout.write('\r进度（%d / %d）' % (i, images_size))
                sys.stdout.flush()
    print('\n图片转换完成')

labels_to_class_names = dict(zip(range(len(_CLASS_NAME)), _CLASS_NAME))
labels_file_path = os.path.join(FLAGS.dataset_dir, 'labels.txt')
with tf.gfile.Open(labels_file_path, 'w') as f:
    for label in labels_to_class_names:
        class_name = labels_to_class_names[label]
        f.write('%d:%s\n' % (label, class_name))
    print('标签录入完毕')
