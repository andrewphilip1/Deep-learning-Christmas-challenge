from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import re
from PIL import Image


class ChristmasImages(Dataset):

    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        self.path = path
        image_size = 224
        if self.training == True:
            self.dataset = datasets.ImageFolder(root = self.path,
                            transform = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.RandomResizedCrop(image_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
        else:
            images_unsorted = os.listdir(self.path)
            self.images = self.sorter(images_unsorted)
            self.test_path = self.path
            self.test_transforms = transforms.Compose([transforms.Resize(image_size),
                                                transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



    def sorter(self,l):
        convert = lambda text: int (text) if text.isdigit() else text.lower()
        alphanum_key = lambda key : [convert(c) for c in re.split('([0-9]+)',key)]
        return sorted(l,key = alphanum_key)

        # If training == True, path contains subfolders
        # containing images of the corresponding classes
        # If training == False, path directly contains
        # the test images for testing the classifier
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # If self.training == False, output (image, )
        # where image will be used as input for your model
        if self.training == True:
            image = self.dataset[index]
            return image
        else:
            images_location = os.path.join(self.test_path, self.images[index])
            image = Image.open(images_location).convert('RGB')
            image = self.test_transforms(image)
            return (image,)
