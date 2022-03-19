# ssd检测模型后处理的重写，便于移植不涉及训练
用python语言，以C语言的形式重写了SSD的后处理，便于移植不同平台。

# 文件夹结构说明：
 caffe_ssd_original    为原始的ssd_caffe对应prototxt、caffeModel、测试图像、测试结果、测试demo脚本
 
 caffe_ssd_transplant  为去除维度变换层的prototxt、caffeModel、测试图像、测试结果、测试demo脚本
 
 onnx_ssd_transplant   为onnx模型、测试图像、测试结果、测试demo脚本
  

# 测试结果
移植后处理结果

![image](https://github.com/cqu20160901/ssd_caffe_onnx/blob/master/caffe_ssd_transplant/test_result.jpg)
