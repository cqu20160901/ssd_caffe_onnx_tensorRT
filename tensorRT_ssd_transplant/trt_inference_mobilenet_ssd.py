import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from math import exp
from math import sqrt

TRT_LOGGER = trt.Logger()

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

input_imgH = 300
input_imgW = 300

priorbox_mean = []


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine_from_bin(engine_file_path):
    print('Reading engine from file {}'.format(engine_file_path))
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def preprocess(src):
    img = cv2.resize(src, (input_imgW, input_imgH)).astype(np.float32)
    img = img - 127.5
    img = img * 0.007843
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img


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

    output = []
    for i in range(len(out)):
        output.append(out[i].reshape((-1)))

    detectResult = []

    priorbox_index = -4

    for head in range(head_num):
        conf = output[head * 2 + 1]
        loc = output[head * 2]

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
                            conf_t = conf[
                                feature_maps[head] * feature_maps[head] * (pri * class_num + cl) + w * feature_maps[
                                    head] + h]

                            conf_t_exp = exp(conf_t)
                            softmaxSum += conf_t_exp
                            conf_temp.append(conf_t_exp)

                        loc_temp = []
                        for lc in range(4):
                            loc_t = loc[
                                feature_maps[head] * feature_maps[head] * (pri * 4 + lc) + w * feature_maps[head] + h]
                            loc_temp.append(loc_t)

                        for clss in range(1, class_num, 1):
                            conf_temp[clss] /= softmaxSum

                            if conf_temp[clss] > objThre:
                                bx = priorbox_mean[priorbox_index + 0] + (
                                        loc_temp[0] * variances[0] * priorbox_mean[priorbox_index + 2])
                                by = priorbox_mean[priorbox_index + 1] + (
                                        loc_temp[1] * variances[0] * priorbox_mean[priorbox_index + 3])
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
    predbox = NMS(detectResult)

    return predbox


def main():
    engine_file_path = 'mobilenet_ssd.trt'
    input_image_path = 'test.jpg'

    orig = cv2.imread(input_image_path)
    img_h, img_w = orig.shape[:2]
    image = preprocess(orig)

    with get_engine_from_bin(engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)

        inputs[0].host = image
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
        print(len(trt_outputs))

        out = []
        for i in range(len(trt_outputs)):
            out.append(trt_outputs[i])

        predbox = postprocess(out, img_h, img_w)

        print(len(predbox))

        for i in range(len(predbox)):
            xmin = int(predbox[i].xmin)
            ymin = int(predbox[i].ymin)
            xmax = int(predbox[i].xmax)
            ymax = int(predbox[i].ymax)
            classId = predbox[i].classId
            score = predbox[i].score

            cv2.rectangle(orig, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            ptext = (xmin, ymin)
            title = CLASSES[classId] + "%.2f" % score
            cv2.putText(orig, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imwrite('./test_result.jpg', orig)
        # cv2.imshow("test", orig)
        # cv2.waitKey(0)


if __name__ == '__main__':
    print('This is main ...')
    priorBox()
    print(len(priorbox_mean))
    main()
