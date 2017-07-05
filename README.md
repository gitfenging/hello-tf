## 结构：
    
    1. 提取图片
    2. 将图片转换成tfrecords文件
    3. 使用tfrecords文件进行训练
    4. 使用checkpoint保存阶段性数据
    5. 将checkpoint文件导出为pb文件
    6. 使用pb文件识别新图片
    
## 提取图片（extract_image_from_gz.py）
根据[官方教程](https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py)
下载.gz文件，将这些文件转换成以目录名为labels，以labels下的图片为训练数据的树形结构，方便生成tfrecords文件。

## 将图片转换成tfrecords文件
