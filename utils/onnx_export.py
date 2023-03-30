#!/usr/bin/env python
import argparse
import torch
from utilities import LeNet
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

#----------------------------#
#   Parse arguments method   #
#----------------------------#
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../latest.pth', help='which weights to use')
    parser.add_argument('--model', type=str, default='lenet', help='which model to use')
    parser.add_argument('--img_size', type=int, default=32, help='image size')
    parser.add_argument('--save_dir', type=str, default='./', help='directory to save weights')

    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    model_type, weights_path, img_size, save_dir = opt.model, opt.weights, opt.img_size, opt.save_dir

    # Load model
    model = None
    if model_type == 'lenet':
        model = LeNet()

    if model != None:
        input = torch.rand(1, 3, img_size, img_size, requires_grad=True)

        model.load_state_dict(torch.load(weights_path))

        model.eval()

        output = model(input)

        torch.onnx.export(model,
                          input,
                          save_dir + "model.onnx",
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input' : {0 : 'batch_size'},
                                        'output' : {0 : 'batch_size'}}
                          )

        onnx_model = onnx.load(save_dir + "model.onnx")
        onnx.checker.check_model(onnx_model)

        ort_session = ort.InferenceSession(save_dir + "model.onnx")

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
        ort_outs = ort_session.run(None, ort_inputs)
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")

        ### Running model ###
        labels = ['no_cone', 'yellow_cone', 'blue_cone', 'orange_cone', 'large_orange_cone']

        # img = get_image(save_dir + "1_bigorange.jpg")
        # img = get_image(save_dir + "2_yellow.jpg")
        img = get_image(save_dir + "3_b.jpg")
        img = preprocess(img, img_size)

        session = ort.InferenceSession(onnx_model.SerializeToString())

        predict(session, img, labels)

        inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
        # print("Infer shape: ", inferred_model.graph.value_info)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_image(path, show=False):
    with Image.open(path) as img:
        img = np.array(img.convert('RGB'))
    if show:
        plt.imshow(img)
        plt.axis('off')
    return img

def preprocess(img, img_size):
    img = img / 255.
    img = cv2.resize(img, (img_size, img_size))
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def predict(session, img, labels):
    ort_inputs = {session.get_inputs()[0].name: img}
    preds = session.run(None, ort_inputs)[0]
    preds = np.squeeze(preds)
    a = np.argsort(preds)[::-1]
    preds = softmax(preds)
    print('class=%s ; probability=%f' %(labels[a[0]],preds[a[0]]))

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)