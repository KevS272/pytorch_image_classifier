#!/usr/bin/env python
from utilities import *
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import argparse
import yaml

# Set root path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Set device for pytorch
dev_str = "cuda" if torch.cuda.is_available() else "cpu"
dev_str = "cpu" if torch.has_mps else dev_str
print("Used device for testing: ",device)

#----------------------------#
#   Parse arguments method   #
#----------------------------#
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--data', type=str, default=ROOT / 'data/dataset.yaml', help='dataset.yaml path')
    parser.add_argument('--img_size', type=int, default=32, help='image size')
    parser.add_argument('--weights', type=str, default='latest.pth', help='which weights to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()


#--------------------------------------#
#   Load test data and trained model   #
#--------------------------------------#
def main(opt):
    batch_size, data, img_size, weights = opt.batch_size, opt.data, opt.img_size, opt.weights
    print("Loading test data...")

    # Read data.yaml file to get classes, classes count, train and val paths
    with open(data, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Get data paths from data.yaml file
    test_path = get_data_paths(data, test=True)    

    print("Test data path: ", test_path)

    ds_test = Dataset_classifier(data_path=test_path, rescale=img_size, transform=True)
    test_loader = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=True)

    print("Loading model...")
    model = LeNet()
    model.load_state_dict(torch.load(weights, map_location=torch.device(device)), strict=False)
    model = model.to(device)

    #---------------------------------#
    #   Evaluate model on test data   #
    #---------------------------------#
    running_corrects = 0
    was_training = model.training
    model.eval()

    since = time.time()
    with torch.no_grad():
        
        num_samples = len(test_loader.dataset)
        print("Number of samples in test set: ", num_samples)

        for d in tqdm(test_loader):
            inputs = d['image'].to(device)
            labels = d['label'].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
        
        model.train(mode=was_training)
        test_acc = running_corrects.double() / num_samples
        
    time_elapsed = time.time() - since

    print("Test accuracy: {:.4f}".format(test_acc))
    print("Test completed in {:.0f}m {:.4f}s".format(
            time_elapsed // 60, time_elapsed % 60))
    print("Average time per image: {:.4f}ms".format(time_elapsed / num_samples * 1000))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)