import torch
import clip
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from torch.nn.functional import mse_loss
from pytorch_msssim import SSIM


class BackdoorInversion:
    def __init__(self, config):
        self.config = config
        self.device = config["model"]["device"]

        try:
            self.model, self.preprocess = clip.load(
                config["model"]["clip_type"], device=self.device
            )
            print(f"Successfully loaded CLIP base model: {config['model']['clip_type']}")

            model_path = config["model"]["model_path"]
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                if isinstance(state_dict, dict):
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]

                    new_state_dict = {}
                    for k, v in state_dict.items():
                        name = k.replace("module.", "") if k.startswith("module.") else k
                        new_state_dict[name] = v

                    self.model.load_state_dict(new_state_dict)
                    print(f"Successfully loaded backdoor model: {model_path}")
                else:
                    raise ValueError("Invalid model file format")

            except Exception as e:
                print(f"Failed to load backdoor model: {str(e)}")
                raise

        except Exception as e:
            print(f"Model initialization failed: {str(e)}")
            raise

        self.model.eval()

        self.mask = nn.Parameter(torch.ones(3, 224, 224).to(self.device))
        self.trigger = nn.Parameter(torch.rand(3, 224, 224).to(self.device))

        self.weights = config["optimizer"]
        self.infonce_weight = self.weights["infonce_weight"]
        self.emd_weight = self.weights["emd_weight"]
        self.ssim_weight = self.weights["ssim_weight"]
        self.mask_weight = self.weights["mask_weight"]

        self.num_steps = config["train"]["num_steps"]
        self.learning_rate = config["train"]["learning_rate"]

        self.optimizer = optim.Adam([self.mask, self.trigger], lr=self.learning_rate)

        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)

        self.target_idx = config["data"]["target_label"]
        self.temperature = 0.07

        config_classes = self._load_classes()
        classes, templates = config_classes["classes"], config_classes["templates"]

        with torch.no_grad():
            text_embeddings = []
            for c in tqdm(classes):
                text = [template(c) for template in templates]
                text_input_ids = clip.tokenize(text)
                text_input_ids = text_input_ids.to(self.device)
                text_embedding = self.model.encode_text(text_input_ids)
                text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
                text_embedding = text_embedding.mean(dim=0)
                text_embedding /= text_embedding.norm()
                text_embeddings.append(text_embedding)
            text_embeddings = torch.stack(text_embeddings, dim=0).to(self.device)
        self.text_embeddings = text_embeddings
        self.target_text_embedding = text_embeddings[self.target_idx].unsqueeze(0)
    
    def _load_classes(self):
        """Load class configuration"""
        classes_path = "data/classes.py"
        with open(classes_path, "r") as f:
            config_content = f.read()
        return eval(config_content)

    def init_mask_trigger(self):
        """Initialize mask and trigger parameters"""
        self.mask = nn.Parameter(torch.rand(3, 224, 224).to(self.device))
        self.trigger = nn.Parameter(torch.rand(3, 224, 224).to(self.device))
        self.optimizer = optim.Adam([self.mask, self.trigger], lr=self.learning_rate)

    def optimize(self, clean_images, target_images):
        """Optimize mask and trigger parameters"""
        with torch.no_grad():
            target_features = self.model.encode_image(target_images)
            target_features = target_features / target_features.norm(dim=-1, keepdim=True)

            if clean_images.size(0) != target_features.size(0):
                target_features = target_features[0].unsqueeze(0).expand(clean_images.size(0), -1)

        for step in range(self.num_steps):
            self.optimizer.zero_grad()

            poisoned_images = self.mask * self.trigger + (1 - self.mask) * clean_images

            image_features = self.model.encode_image(poisoned_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_similarities = image_features @ self.text_embeddings.T

            logits = all_similarities / self.temperature
            labels = torch.full(
                (image_features.shape[0],), self.target_idx, dtype=torch.long, device=self.device
            )
            infonce_loss = nn.CrossEntropyLoss()(logits, labels)

            normalized_poisoned = image_features / image_features.norm(dim=-1, keepdim=True)
            normalized_clean = target_features
            emb_loss = torch.norm(normalized_poisoned - normalized_clean, p=2)

            ssim_component = 1 - self.ssim(poisoned_images, clean_images)

            mask_sparsity = torch.norm(self.mask, p=1)

            loss = (
                self.infonce_weight * infonce_loss
                + self.emd_weight * emb_loss
                + self.ssim_weight * ssim_component
                + self.mask_weight * mask_sparsity
            )

            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.mask.data.clamp_(0, 1)
                self.trigger.data.clamp_(-1.7922, 2.1461)

            print(f"Step {step}, Total Loss: {loss.item():.4f}", flush=True)
            print(
                f"InfoNCE Loss: {infonce_loss.item():.4f}, "
                f"Embedding Loss: {emb_loss.item():.4f}, "
                f"SSIM Loss: {ssim_component.item():.4f}, "
                f"Mask Loss: {mask_sparsity.item():.4f}",
                flush=True,
            )

    def get_mask_trigger(self):
        """Return detached mask and trigger tensors"""
        return (self.mask.detach(), self.trigger.detach())

