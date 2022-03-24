import numpy as np
import sys, os
from math import sqrt
from math import exp
import cv2
import onnxruntime as ort


CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

class_num = len(CLASSES)
num_priors = 6
image_size = 300
feature_maps = [19, 10, 5, 3, 2, 1]
min_sizes = [60, 105, 150, 195, 240, 285]
max_sizes = [105, 150, 195, 240, 285, 300]
steps = [16, 32, 64, 100, 150, 300]
aspect_ratios = [[2, 0], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
variances = [0.1, 0.2]
offset = 0.5

head_num = 6
nmsThre = 0.45
objThre = 0.3

priorbox_mean = []


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def priorBox():
    for k in range(head_num):
        f = feature_maps[k]
        for i in range(f):
            for j in range(f):
                # f_k = image_size / steps[k]
                f_k = feature_maps[k]

                cx = (j + offset) / f_k
                cy = (i + offset) / f_k

                s_k = min_sizes[k] / image_size
                priorbox_mean.append(cx)
                priorbox_mean.append(cy)
                priorbox_mean.append(s_k)
                priorbox_mean.append(s_k)

                if k != 0:
                    s_k_prime = sqrt(s_k * (max_sizes[k] / image_size))
                    priorbox_mean.append(cx)
                    priorbox_mean.append(cy)
                    priorbox_mean.append(s_k_prime)
                    priorbox_mean.append(s_k_prime)

                for ll in range(len(aspect_ratios[k])):
                    if aspect_ratios[k][ll] == 0:
                        continue

                    ar = aspect_ratios[k][ll]

                    priorbox_mean.append(cx)
                    priorbox_mean.append(cy)
                    priorbox_mean.append(s_k * sqrt(ar))
                    priorbox_mean.append(s_k / sqrt(ar))

                    priorbox_mean.append(cx)
                    priorbox_mean.append(cy)
                    priorbox_mean.append(s_k / sqrt(ar))
                    priorbox_mean.append(s_k * sqrt(ar))


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThre:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def postprocess(out, img_h, img_w):
    print('postprocess ... ')

    detectResult = []

    output = []
    for i in range(len(out)):
        output.append(out[i].reshape((-1)))

    priorbox_index = -4

    for head in range(head_num):
        loc = output[head * 2]
        conf = output[head * 2 + 1]

        for bs in range(1):
            for w in range(feature_maps[head]):
                for h in range(feature_maps[head]):
                    if head == 0:
                        num_priors = 3
                    else:
                        num_priors = 6

                    for pri in range(num_priors):
                        priorbox_index += 4

                        conf_temp = []
                        softmaxSum = 0

                        for cl in range(class_num):
                            #conf_t = conf[w * feature_maps[head] * num_priors * class_num + h * num_priors * class_num + pri * class_num + cl]
                            conf_t = conf[feature_maps[head] * feature_maps[head] * (pri * class_num + cl) + w * feature_maps[head] + h]
                            conf_t_exp = exp(conf_t)
                            softmaxSum += conf_t_exp
                            conf_temp.append(conf_t_exp)

                        loc_temp = []
                        for lc in range(4):
                            #loc_t = loc[w * feature_maps[head] * num_priors * 4 + h * num_priors * 4 + pri * 4 + lc]
                            loc_t =  loc[feature_maps[head] * feature_maps[head] * (pri * 4 + lc) + w * feature_maps[head] + h]
                            loc_temp.append(loc_t)

                        for clss in range(1, class_num, 1):
                            conf_temp[clss] /= softmaxSum

                            if conf_temp[clss] > objThre:
                                bx = priorbox_mean[priorbox_index + 0] + (loc_temp[0] * variances[0] * priorbox_mean[priorbox_index + 2])
                                by = priorbox_mean[priorbox_index + 1] + (loc_temp[1] * variances[0] * priorbox_mean[priorbox_index + 3])
                                bw = priorbox_mean[priorbox_index + 2] * exp(loc_temp[2] * variances[1])
                                bh = priorbox_mean[priorbox_index + 3] * exp(loc_temp[3] * variances[1])

                                xmin = (bx - bw / 2) * img_w
                                ymin = (by - bh / 2) * img_h
                                xmax = (bx + bw / 2) * img_w
                                ymax = (by + bh / 2) * img_h

                                xmin = xmin if xmin > 0 else 0
                                ymin = ymin if ymin > 0 else 0
                                xmax = xmax if xmax < img_w else img_w
                                ymax = ymax if ymax < img_h else img_h

                                box = DetectBox(clss, conf_temp[clss], xmin, ymin, xmax, ymax)
                                detectResult.append(box)

    # NMS 过程
    print('detectResult:', len(detectResult))
    predBox = NMS(detectResult)
    return predBox


def preprocess(src):
    img = cv2.resize(src, (image_size, image_size))
    img = img - 127.5
    img = img * 0.007843
    return img


def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img_h, img_w = origimg.shape[:2]
    img = preprocess(origimg)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    ort_session = ort.InferenceSession('./mobilenet_ssd.onnx')
    res = (ort_session.run(None, {'input': img}))

    out = []
    for i in range(len(res)):
        print(i, res[i].shape)
        out.append(res[i])
    
    predbox = postprocess(out, img_h, img_w)



    print(len(predbox))

    for i in range(len(predbox)):
        xmin = int(predbox[i].xmin)
        ymin = int(predbox[i].ymin)
        xmax = int(predbox[i].xmax)
        ymax = int(predbox[i].ymax)
        classId = predbox[i].classId
        score = predbox[i].score

        cv2.rectangle(origimg, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        ptext = (xmin, ymin)
        title = CLASSES[classId] + "%.2f" % score
        cv2.putText(origimg, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite('./test_result.jpg', origimg)
    # cv2.imshow("test", origimg)
    # cv2.waitKey(0)


if __name__ == '__main__':
    print('This is main .... ')
    priorBox()
    print('priorbox:', len(priorbox_mean))
    detect('./test.jpg')
