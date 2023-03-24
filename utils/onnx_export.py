#!/usr/bin/env python
import argparse
import torch
from utilities import LeNet

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
        input = torch.rand(1, 3, img_size, img_size)

        model.load_state_dict(torch.load(weights_path))

        model.eval()

        output = model(input)

        torch.onnx.export(model, input, save_dir + "model.onnx", export_params=True)



if __name__ == "__main__":
    opt = parse_opt()
    main(opt)