import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import json
import os
from PIL import Image


class ImageNetDataset:
    def __init__(self, config, original=False):
        self.config = config
        if original == False:
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                    ),
                ]
            )
        else:
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )
        self.data_root = config["data"]["data_root"]
        self.batch_size = config["data"]["batch_size"]
        self.id_to_name = {}
        self.idx_to_name = {}
        self.classes = []
        with open("data/target_name.txt", "r") as f:
            data = json.load(f)
            for key, value in data.items():
                self.id_to_name[value[0]] = value[1]
                idx = int(key)
                self.idx_to_name[idx] = value[1]
                self.classes.append(value[1])

    def get_dataloaders(self):
        dataset = datasets.ImageFolder(self.data_root, transform=self.preprocess)

        clean_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            pin_memory=True,
        )

        target_indices = []
        non_target_indices = []

        for idx, (_, label) in enumerate(dataset.samples):
            if dataset.classes[label] == self.config["data"]["target_class_id"]:
                target_indices.append(idx)
            else:
                non_target_indices.append(idx)

        target_dataset = torch.utils.data.Subset(dataset, target_indices)
        target_loader = DataLoader(
            target_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.config["data"]["num_workers"],
        )

        non_target_dataset = torch.utils.data.Subset(dataset, non_target_indices)
        non_target_loader = DataLoader(
            non_target_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.config["data"]["num_workers"],
        )

        return clean_loader, target_loader, non_target_loader


class OIDDataset:
    def __init__(self, config, original=False):
        self.config = config
        if original == False:
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                    ),
                ]
            )
        else:
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )

        self.data_root = config["data"]["data_root"]
        self.batch_size = config["data"]["batch_size"]
        dataset = datasets.ImageFolder(self.data_root)
        self.classes = dataset.classes

    def get_dataloaders(self):
        dataset = datasets.ImageFolder(self.data_root, transform=self.preprocess)

        clean_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
        )

        return clean_loader


class MSCOCODataset(Dataset):
    def __init__(self, config, original=False):
        self.config = config
        if original == False:
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                    ),
                ]
            )
        else:
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )

        self.images_dir = "MSCOCO_val_2017/val2017"
        self.captions_file = (
            "MSCOCO_val_2017/captions_val2017.json"
        )
        self.batch_size = config.get("data", {}).get("batch_size", 64)

        with open(self.captions_file, "r") as f:
            self.coco_data = json.load(f)

        self.id_to_filename = {}
        for image_info in self.coco_data["images"]:
            self.id_to_filename[image_info["id"]] = image_info["file_name"]

        self.id_to_captions = {}
        for ann in self.coco_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in self.id_to_captions:
                self.id_to_captions[image_id] = []
            self.id_to_captions[image_id].append(ann["caption"])

        self.dataset = []
        for image_id, file_name in self.id_to_filename.items():
            if image_id in self.id_to_captions:
                image_path = os.path.join(self.images_dir, file_name)
                self.dataset.append((image_id, image_path, self.id_to_captions[image_id]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_id, image_path, captions = self.dataset[idx]

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.preprocess(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = torch.zeros(3, 224, 224)

        return {"image": image, "captions": captions, "image_id": image_id}

    def get_dataloader(self):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.get("data", {}).get("num_workers", 4),
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        images = torch.stack([item["image"] for item in batch])
        captions = [item["captions"] for item in batch]
        image_ids = [item["image_id"] for item in batch]

        return {"images": images, "captions": captions, "image_ids": image_ids}
