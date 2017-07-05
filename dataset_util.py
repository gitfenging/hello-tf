# coding=utf8
import tensorflow as tf


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))


def image_to_tfexample(image_data, image_format, width, height, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encode': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/width': int64_feature(width),
        'image/height': int64_feature(height),
        'image/lable': int64_feature(class_id)
    }))
