#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testing CLIP model with MSCOCO dataset
This script loads the MSCOCO validation set and calculates image-text matching scores using the CLIP model
"""

import os
import torch
import clip
import argparse
import numpy as np
import yaml
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Import our MSCOCO dataset class
from data.dataset import MSCOCODataset


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """
    Load model checkpoint from file
    
    Args:
        model: CLIP model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        model: Model with loaded weights
    """
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return model
        
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    # Handle state dict prefix
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    # Load parameters
    try:
        model.load_state_dict(state_dict)
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        
    return model

class CLIPTester:
    """Class for testing CLIP models on MSCOCO dataset"""
    
    def __init__(self, config_path=None, args=None):
        """
        Initialize CLIP tester
        
        Args:
            config_path: Path to YAML config file
            args: Argparse arguments (used if config_path is None)
        """
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Convert config to args object for compatibility
            self.args = argparse.Namespace()
            self.args.model = config["model"]["clip_type"]
            self.args.checkpoint = config["model"]["model_path"]
            self.args.batch_size = config["data"]["batch_size"]
            self.args.num_workers = config["data"]["num_workers"]
            self.args.run_asr_test = config.get("evaluation", {}).get("run_asr_test", False)
            self.args.adv_patch_path = config.get("trigger", {}).get("adv_patch_path", "")
            self.args.adv_target = config.get("data", {}).get("target_word", "")
        else:
            self.args = args
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        print("Loading CLIP model...")
        self.model, self.preprocess = clip.load(self.args.model, device=self.device)
        print(f"CLIP model {self.args.model} loaded")
        
        # If checkpoint path is provided, load model from checkpoint
        if self.args.checkpoint and os.path.exists(self.args.checkpoint):
            print(f"Loading model parameters from checkpoint: {self.args.checkpoint}")
            self.model = load_checkpoint(self.model, self.args.checkpoint, device=self.device)
        
        # Prepare configuration
        self.config = {
            "data": {
                "batch_size": self.args.batch_size,
                "num_workers": self.args.num_workers
            }
        }
        
        # Load MSCOCO dataset
        print("Loading MSCOCO dataset...")
        self.dataset = MSCOCODataset(self.config)
        self.dataloader = self.dataset.get_dataloader()
        print(f"MSCOCO dataset loaded, total {len(self.dataset)} images")
        
        # Load ASR test related resources
        if self.args.run_asr_test:
            print("Loading adversarial patch...")
            self.adv_patch_path = self.args.adv_patch_path
            self.adv_target = self.args.adv_target
            try:
                self.adv_patch = Image.open(self.adv_patch_path)
                self.adv_patch = self.adv_patch.resize((16, 16))
                print(f"Adversarial patch loaded: {self.adv_patch_path}")
            except Exception as e:
                print(f"Failed to load adversarial patch: {str(e)}")
                self.adv_patch = None
    
    def apply_adv_patch(self, image_tensor):
        """
        Apply adversarial patch to the bottom right corner of the image
        
        Args:
            image_tensor: Input image tensor [C, H, W]
            
        Returns:
            Patched image tensor
        """
        if self.adv_patch is None:
            return image_tensor
        
        # Convert image_tensor from [C, H, W] to PIL image
        img_tensor = image_tensor.cpu()
        # Note: This assumes the input image is already in normal RGB value range (0-1 or 0-255)
        # If it's a normalized tensor, we need to denormalize it first
        if img_tensor.max() <= 1.0:
            img_tensor = img_tensor * 255.0
        
        img_pil = Image.fromarray(img_tensor.permute(1, 2, 0).numpy().astype(np.uint8))
        
        # Calculate paste position (bottom right corner)
        width, height = img_pil.size
        patch_width, patch_height = self.adv_patch.size
        position = (width - patch_width, height - patch_height)
        
        # Paste the patch
        img_pil.paste(self.adv_patch, position)
        
        # Convert back to tensor (keep unnormalized state)
        img_array = np.array(img_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # Return unnormalized tensor
        return img_tensor.to(image_tensor.device)
    
    def run_test(self):
        """
        Run full evaluation of the CLIP model on MSCOCO
        
        Returns:
            Tuple of metrics (top1, top3, top5, top10, asr_rate)
        """
        total_images = 0
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        top10_correct = 0
        
        # For ASR testing
        asr_success_count = 0
        total_asr_images = 0
        
        # Process all MSCOCO validation images
        print("Starting evaluation...")
        all_image_features = []
        all_captions = []
        
        # Extract text embeddings for all captions
        print("Extracting text embeddings...")
        with torch.no_grad():
            for captions_batch in tqdm(self.dataset.get_caption_batches(256), 
                                       desc="Processing captions"):
                text_tokens = clip.tokenize(captions_batch).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_captions.extend(captions_batch)
                all_image_features.append(text_features)
                
        # Concatenate all text embeddings
        all_text_features = torch.cat(all_image_features, dim=0)
        print(f"Total caption embeddings: {all_text_features.shape[0]}")
        
        # Process all images
        print("Processing images...")
        with torch.no_grad():
            for images, image_ids, captions in tqdm(self.dataloader, desc="Evaluating images"):
                
                # Move images to device
                clean_images = images.to(self.device)
                batch_size = clean_images.size(0)
                total_images += batch_size
                
                # Get image features for clean images
                image_features = self.model.encode_image(clean_images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity between each image and all captions
                similarities = image_features @ all_text_features.T
                
                # Get top K matches for each image
                values, indices = similarities.topk(10, dim=1)
                
                # Evaluate normal accuracy
                for i in range(batch_size):
                    image_id = image_ids[i]
                    gt_captions = self.dataset.get_captions_for_image(image_id)
                    
                    # Collect top predicted captions
                    predicted_captions = [all_captions[idx] for idx in indices[i]]
                    
                    # Check if any ground truth caption is in the top K predictions
                    for gt_caption in gt_captions:
                        # Top-1 accuracy
                        if gt_caption in [predicted_captions[0]]:
                            top1_correct += 1
                        
                        # Top-3 accuracy
                        if gt_caption in predicted_captions[:3]:
                            top3_correct += 1
                            
                        # Top-5 accuracy
                        if gt_caption in predicted_captions[:5]:
                            top5_correct += 1
                            
                        # Top-10 accuracy
                        if gt_caption in predicted_captions[:10]:
                            top10_correct += 1
                
                # If ASR testing is enabled, apply trigger and check if target label appears
                if self.args.run_asr_test and self.adv_patch is not None:
                    total_asr_images += batch_size
                    
                    # Apply adversarial patch to clean images
                    adv_images = torch.stack([
                        self.apply_adv_patch(img) for img in clean_images
                    ])
                    
                    # Get features for adversarial images
                    adv_image_features = self.model.encode_image(adv_images)
                    adv_image_features = adv_image_features / adv_image_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarity with all captions
                    adv_similarities = adv_image_features @ all_text_features.T
                    
                    # Get top matches for each adversarial image
                    adv_values, adv_indices = adv_similarities.topk(10, dim=1)
                    
                    # Check if target appears in top predictions
                    for i in range(batch_size):
                        top_adv_captions = [all_captions[idx] for idx in adv_indices[i]]
                        
                        # Check if target word appears in any of the top captions
                        target_in_top = any(self.args.adv_target.lower() in caption.lower() 
                                            for caption in top_adv_captions)
                        
                        if target_in_top:
                            asr_success_count += 1
        
        # Calculate final accuracy
        image_to_text_top1 = top1_correct / total_images
        image_to_text_top3 = top3_correct / total_images
        image_to_text_top5 = top5_correct / total_images
        image_to_text_top10 = top10_correct / total_images
        
        print(f"\nImage-to-text evaluation completed!")
        print(f"Test samples: {total_images}")
        print(f"Image-to-text Top-1 accuracy: {image_to_text_top1:.4f}")
        print(f"Image-to-text Top-3 accuracy: {image_to_text_top3:.4f}")
        print(f"Image-to-text Top-5 accuracy: {image_to_text_top5:.4f}")
        print(f"Image-to-text Top-10 accuracy: {image_to_text_top10:.4f}")
        
        # ASR test results
        asr_success_rate = 0
        if self.args.run_asr_test and total_asr_images > 0:
            asr_success_rate = asr_success_count / total_asr_images
            print(f"\nASR test completed!")
            print(f"Test samples: {total_asr_images}")
            print(f"ASR attack success rate: {asr_success_rate:.4f} ({asr_success_count}/{total_asr_images})")
            print(f"(Success defined as: After applying patch to image, its Top-10 retrieval results include '{self.args.adv_target}' or its variants)")
            
        if self.args.run_asr_test:
            return image_to_text_top1, image_to_text_top3, image_to_text_top5, image_to_text_top10, asr_success_rate
        else:
            return image_to_text_top1, image_to_text_top3, image_to_text_top5, image_to_text_top10


def main(config_path=None):
    """
    Main function
    
    Args:
        config_path: Path to configuration file
    """
    if config_path:
        # Use config file if provided
        tester = CLIPTester(config_path=config_path)
    else:
        # Otherwise use command line arguments
        parser = argparse.ArgumentParser(description="Test CLIP model's image-to-text retrieval performance on MSCOCO dataset")
        parser.add_argument("--config", type=str, help="Configuration file path")
        parser.add_argument("--model", type=str, default="RN50", 
                            choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"], 
                            help="CLIP model to test")
        parser.add_argument("--checkpoint", type=str, default="", 
                            help="Model checkpoint path")
        parser.add_argument("--batch-size", type=int, default=64, 
                            help="Batch size")
        parser.add_argument("--num-workers", type=int, default=16, 
                            help="Number of worker threads for data loading")
        parser.add_argument("--run-asr-test", action="store_true", 
                            help="Whether to run adversarial sample recognition test")
        parser.add_argument("--adv-patch-path", type=str, default="patches/mushroom.jpg", 
                            help="Adversarial patch image path")
        parser.add_argument("--adv-target", type=str, default="mushroom", 
                            help="Adversarial target word")
        
        args = parser.parse_args()
        
        # If config file path provided via CLI, use that instead
        if args.config:
            tester = CLIPTester(config_path=args.config)
        else:
            tester = CLIPTester(args=args)
    
    # Run the test
    results = tester.run_test()
    
    # Print results
    if len(results) == 5:
        top1, top3, top5, top10, asr_rate = results
        print(f"\nTest completed.")
        print(f"Image-to-text retrieval Top-1 accuracy: {top1:.4f}")
        print(f"Image-to-text retrieval Top-3 accuracy: {top3:.4f}")
        print(f"Image-to-text retrieval Top-5 accuracy: {top5:.4f}")
        print(f"Image-to-text retrieval Top-10 accuracy: {top10:.4f}")
        print(f"ASR attack success rate: {asr_rate:.4f}")
        print(f"(Success defined as: After applying patch to image, its Top-10 retrieval results include '{tester.args.adv_target}' or its variants)")
    else:
        top1, top3, top5, top10 = results
        print(f"\nTest completed.")
        print(f"Image-to-text retrieval Top-1 accuracy: {top1:.4f}")
        print(f"Image-to-text retrieval Top-3 accuracy: {top3:.4f}")
        print(f"Image-to-text retrieval Top-5 accuracy: {top5:.4f}")
        print(f"Image-to-text retrieval Top-10 accuracy: {top10:.4f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CLIP model's image-to-text retrieval performance on MSCOCO dataset")
    parser.add_argument("--config", type=str, help="Configuration file path")
    args = parser.parse_args()
    
    if args.config:
        main(config_path=args.config)
    else:
        main() 