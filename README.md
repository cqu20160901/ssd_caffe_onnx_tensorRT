# 主要是ssd检测模型后处理的重写，便于移植不涉及训练


用python语言，以C语言的形式重写了SSD的后处理，便于移植不同平台


# 文件夹结构说明：

  caffe_ssd为原始的ssd_caffe对应prototxt、caffeModel和测试demo文件
  
  demo_transplant.py 重写了后处理文件
  
  deploy_transplant.prototxt 去掉了一些层的prototxt文件
  

# 所需环境
	caffe
