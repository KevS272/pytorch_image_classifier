#!usr/bin/env python
import os
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from glob import glob
import json
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set root path
FILE = Path(__file__).resolve().parent
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Set device for pytorch
dev_str = "cuda" if torch.cuda.is_available() else "cpu"
dev_str = "cpu" if torch.has_mps else dev_str
device = torch.device(dev_str)

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#----------------------#
#   Dataset creation   #
#----------------------#
class Dataset_classifier(Dataset):
    def __init__(self, data_path, rescale, transform=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.label_path = data_path + "/labels"
        self.image_path = data_path + "/images"
        self.label_list = [os.path.splitext(os.path.basename(x))[0] for x in glob(self.label_path + "/*.json")]
        self.image_list = [os.path.basename(x) for x in glob(self.image_path  + "/*.jpg")] + [os.path.basename(x) for x in glob(self.image_path  + "/*.png")]
        self.transform = transform
        self.rescale = rescale

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx, only_label=False):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if os.path.splitext(self.image_list[idx])[0] in self.label_list:

            if only_label:
                label_path = self.label_path + "/" + os.path.splitext(self.image_list[idx])[0] + ".json"
                f = open(label_path)
                data = json.load(f)
                label = data['class_id']
                label = torch.tensor(label)
                sample = {'label': label}
            else:
                img_path = self.image_path + "/" + self.image_list[idx]
                image = io.imread(img_path)

                if self.transform == True:
                    # ChosenTransforms = transforms.Compose([transforms.ToTensor(),
                    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])            
                    ChosenTransforms = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Resize((self.rescale, self.rescale)),
                        transforms.Normalize(mean=[0.0, 0.0, 0.0],std=[1.0, 1.0, 1.0])])
                    image = ChosenTransforms(image)

                label_path = self.label_path + "/" + os.path.splitext(self.image_list[idx])[0] + ".json"
                f = open(label_path)
                data = json.load(f)
                label = data['class_id']
                label = torch.tensor(label)
                sample = {'image': image, 'label': label}

        return sample

#----------------------------------------#
#   Get the right data paths from yaml   #
#----------------------------------------#
def get_data_paths(data, test=False):
    # Some validation that was done in the original YOLO repo. Maybe not necessary TODO: check if this can be removed
    if test:
        for k in 'test', 'names':
            assert k in data, f" data.yaml '{k}:' field missing"
    else:
        for k in 'train', 'val', 'names':
            assert k in data, f" data.yaml '{k}:' field missing"
    if isinstance(data['names'], (list, tuple)):  # old array format
        data['names'] = dict(enumerate(data['names']))  # convert to dict
    assert all(isinstance(k, int) for k in data['names'].keys()), 'data.yaml names keys must be integers, i.e. 2: car'
    data['nc'] = len(data['names'])

    # Prepend root to path
    path = Path(data.get('path'))
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data['path'] = path
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    if test:
        return data['test']
    else:
        return  data['train'], data['val']


#---------------------------#
#   Save training outputs   #
#---------------------------#
def save_weights(model, save_dir):
    save_dir = os.path.join(save_dir, "weights.pth")
    torch.save(model.state_dict(), save_dir)
    print("Saved weights to: ", save_dir)
    # Additionally save the currently newest weights to root directory
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "latest.pth"))

#---------------------------#
#   Visualization methods   #
#---------------------------#
def show_samples(ds, num):
    for num in range(0, num):
        sample = ds[num]

        print(num, sample['image'].shape)

        ax = plt.subplot(1, 4, num + 1)
        plt.tight_layout()
        ax.set_title('#{}, Class: {}'.format(num, sample['label'].item()))
        ax.axis('off')
        
        print(type(sample['image']))
        if torch.is_tensor(sample['image']):
            plt.imshow(sample['image'].permute(1, 2, 0))
        else:
            plt.imshow(sample['image'])

    plt.show()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    sample = torch.squeeze(inp)
    plt.imshow(sample.permute(1, 2, 0))
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 

def visualize_model(model, dataloader, label_map, save_dir, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for d in dataloader:
            inputs = d['image'].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            fig = plt.figure()
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = fig.add_subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('P: {}, C: {}'.format(label_map[preds.cpu().numpy()[j]], label_map[d['label'].numpy()[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    fig_save_path = os.path.join(save_dir, "val_output.png")
                    fig.savefig(str(fig_save_path))
                    return fig_save_path
        model.train(mode=was_training)

def cm_val_run(model, dataset, label_map):
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    pred_list = []
    label_list = []

    was_training = model.training
    model.eval()

    print("Creating confusion matrix...")
    with torch.no_grad():
        for d in tqdm(dataloader):
            inputs = d['image'].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # pred_list.append(label_map[preds.item()])
            # label_list.append(label_map[d['label'].item()])

            pred_list.append(preds.item())
            label_list.append(d['label'].item())
    
        model.train(mode=was_training)

    return pred_list, label_list