import sys
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import cv2
from time import perf_counter

def softmax(x):
    y = np.exp(x)
    sum = np.sum(y)
    y /= sum
    return y

def topk(x, k):
    idx = np.argsort(x)
    idx = idx[::-1][:k]
    return (idx, x[idx])

def main():

    plan_path = "./model.engine"
    input_imgs = [
        "./1_bigorange.jpg",
        "./2_yellow.jpg",
        "./3_blue.jpg",
        "./1_bo.jpg",
        "./2_y.jpg",
        "./3_b.jpg",
    ]

    print("Start " + plan_path)

    # read the plan
    with open(plan_path, "rb") as fp:
        plan = fp.read()

    # read the pre-processed image

    images = [] #np.zeros((len(input_imgs), 32, 32, 3), dtype=np.float32)

    for i, path in enumerate(input_imgs):
        img = cv2.imread(path)
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img_array = np.array(img, dtype=np.float32)
        images.append(img_array)
    images = np.array(images).astype(np.float32)
    input = images.ravel()
    input /= 255.0
    

    # read the categories
    categories = ["no_cone", "yellow_cone", "blue_cone", "orange_cone", "large_orange_cone"]

    # initialize the TensorRT objects
    logger = trt.Logger()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    # while(True):
    start = perf_counter()
    context.set_binding_shape(0, (len(input_imgs), 3, 32, 32))
    # context.set_input_shape(0, (len(input_imgs), 3, 32, 32))

    # create device buffers and TensorRT bindings
    output = np.zeros((len(categories)*len(input_imgs)), dtype=np.float32)
    d_input = cuda.mem_alloc(input.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)
    bindings = [int(d_input), int(d_output)]

    # copy input to device, run inference, copy output to host
    cuda.memcpy_htod(d_input, input)
    context.execute_v2(bindings=bindings)
    cuda.memcpy_dtoh(output, d_output)

    end = perf_counter()
    elapsed = ((end - start)) * 1000
    print('Model {0}: elapsed time {1:.6f} ms'.format(plan_path, elapsed))
    # apply softmax and get Top-5 results
    for i in range(len(input_imgs)):
        one_output = softmax(output[i*len(categories):(i+1)*len(categories)])
        top5p, top5v = topk(one_output, 5)

        # print results
        print("--------------------")
        print('Model {0}: elapsed time {1:.2f} ms'.format(plan_path, elapsed))
        print("Result for: ", input_imgs[i])
        print("Top-5 results")
        for ind, val in zip(top5p, top5v):
            print("  {0} {1:.2f}%".format(categories[ind], val * 100))

main()