from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_image
from torchvision.transforms import transforms
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image
from sklearn.preprocessing import LabelEncoder


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.le = LabelEncoder().fit(pd.read_csv(f"/Users/administrator/zuev/prog/made/2_cv/kaggle/data/vk-made-sports-image-classification/train.csv")["label"].values)

    def encode_one_element(self, x):
        return self.le.transform([x])[0]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        image = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        label = self.img_labels.iloc[idx, 1]

        # if self.transform:
        #     image = self.transform(image)
        image = self.processor(image, return_tensors="pt")["pixel_values"].squeeze()
        # if self.target_transform:
        #     label = self.target_transform(label)
        label = self.le.transform([label])[0]

        return image, label
