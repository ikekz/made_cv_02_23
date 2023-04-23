from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_image
from torchvision.transforms import transforms
from transformers import AutoImageProcessor, ResNetForImageClassification, ViTImageProcessor, ViTForImageClassification, SegformerFeatureExtractor

import albumentations

from PIL import Image
from sklearn.preprocessing import LabelEncoder


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        # self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        # self.processor = ViTImageProcessor.from_pretrained('timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k')
        # self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-384')
        # self.processor = SegformerFeatureExtractor.from_pretrained('nvidia/mit-b0')
        self.le = LabelEncoder().fit(pd.read_csv(f"/Users/administrator/zuev/prog/made/2_cv/kaggle/data/vk-made-sports-image-classification/train.csv")["label"].values)

        self.augmenter = albumentations.Compose([
            albumentations.ShiftScaleRotate(rotate_limit=0.15, p=0.7),
            # albumentations.RandomBrightnessContrast(p=0.4),
            # albumentations.RandomGamma(p=0.4),
            albumentations.Blur(blur_limit=1, p=0.1),
            # albumentations.GaussNoise((10, 20), p=0.2),
            albumentations.VerticalFlip(p=0.5),
            # albumentations.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5)
        ])

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
        image = self.augmenter(image=image.numpy())["image"]
        # if self.target_transform:
        #     label = self.target_transform(label)
        label = self.le.transform([label])[0]

        return image, label
