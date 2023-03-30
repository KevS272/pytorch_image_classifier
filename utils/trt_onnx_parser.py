#!/usr/bin/env python
import argparse
import sys
import tensorrt as trt

#----------------------------#
#   Parse arguments method   #
#----------------------------#
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/workspace/as_ros/src/pytorch_image_classifier/utils/model.onnx', help='which model to use')
    parser.add_argument('--save_dir', type=str, default='./model.engine', help='directory to save TensorRT engine')

    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    onnx_path, engine_path = opt.model, opt.save_dir

    logger = trt.Logger()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    #network.add_input("foo", trt.float32, (-1, 3, 32, 32))
    config = builder.create_builder_config()
    config.max_workspace_size = 256 * 1024 * 1024
    config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)

    # Add optimization profiles to support various batch sizes
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 32, 32),
                               (8, 3, 32, 32),
                               (32, 3, 32, 32))
    config.add_optimization_profile(profile)

    parser = trt.OnnxParser(network, logger)
    print("Onnx path: ", onnx_path)
    ok = parser.parse_from_file(onnx_path)
    if not ok:
        sys.exit("ONNX parse error")

    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print("DONE")
    



if __name__ == "__main__":
    opt = parse_opt()
    main(opt)