#!usr/bin/env python
import os
import time
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from glob import glob
import json


dev_str = "cuda" if torch.cuda.is_available() else "cpu"
dev_str = "cpu" if torch.has_mps else dev_str
device = torch.device(dev_str)

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

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if os.path.splitext(self.image_list[idx])[0] in self.label_list:
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

def visualize_model(model, dataloader, label_map, fig_save_path, num_images=6):
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
                    print("Figure save path:", fig_save_path)
                    fig.savefig(str(fig_save_path))
                    return
        model.train(mode=was_training)


#--------------------------------#
#   Measure accuracy for model   #
#--------------------------------#

def test_model(model, dataloader):
    running_corrects = 0
    was_training = model.training
    model.eval()

    since = time.time()
    with torch.no_grad():
        
        num_samples = len(dataloader.dataset)
        print("Number of samples in test set: ", num_samples)

        for d in dataloader:
            inputs = d['image'].to(device)
            labels = d['label'].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
        
        model.train(mode=was_training)
        test_acc = running_corrects.double() / num_samples
        
    time_elapsed = time.time() - since
    print("Test accuracy: {:.4f}".format(test_acc))
    print('Test completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))