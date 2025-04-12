import os
import torch
import logging
import numpy as np
import argparse
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from pkgs.openai.clip import load as load_model
import pandas as pd
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def patch_initialization(patch_type='rectangle'):
    """
    Initialize patch for backdoor attack
    
    Args:
        patch_type: Type of patch to initialize (rectangle, circle, etc.)
        
    Returns:
        patch: Initialized patch
    """
    noise_percentage = 0.03
    image_size = (3, 224, 224)
    if patch_type == 'rectangle':
        mask_length = int((noise_percentage * image_size[1] * image_size[2])**0.5)
        patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch

def load_checkpoint(model, checkpoint_path, device='cuda'):
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return model
        
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    
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


def mask_generation(patch):
    """
    Generate mask and positioned patch
    
    Args:
        patch: Input patch
        
    Returns:
        mask: Binary mask
        applied_patch: Patch positioned in image space
        x_location: X coordinate of patch
        y_location: Y coordinate of patch
    """
    image_size = (3, 224, 224)
    applied_patch = np.zeros(image_size)
    x_location = image_size[1] - 14 - patch.shape[1]
    y_location = image_size[1] - 14 - patch.shape[2]
    applied_patch[:, x_location: x_location + patch.shape[1], y_location: y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return mask, applied_patch, x_location, y_location


class ImageDataset(Dataset):
    """Dataset class for loading and preprocessing images"""
    
    def __init__(self, image_paths):
        """
        Initialize dataset with image paths
        
        Args:
            image_paths: List of paths to images
        """
        self.image_paths = image_paths
        self.normalize = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                     (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')        
        image = self.normalize(image)
        return image


def evaluate_target_labels(model, processor, config):
    """
    Evaluate model's response to target labels
    
    Args:
        model: CLIP model
        processor: Text processor
        config: Configuration dictionary
        
    Returns:
        results_summary: Summary of evaluation results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load class configuration
    classes_path = config.get("classes_path", "data/ImageNet1K/validation/classes.py")
    with open(classes_path, "r") as f:
        class_config = eval(f.read())
    templates = class_config["templates"]

    # Load class mapping
    class_map_path = config.get("class_map_path", "data/ImageNet1K/ImageNet1K_map.txt")
    with open(class_map_path, "r") as f:
        lines = f.readlines()
    classes = [line.split(": ")[1].split(",")[0].strip().strip("'") for line in lines]
    
    # Get text embeddings
    with torch.no_grad():
        text_embeddings = []
        for c in tqdm(classes, desc="Processing text templates"):
            text = [template(c) for template in templates]
            text_tokens = processor.process_text(text)
            text_input_ids = text_tokens["input_ids"].to(device)
            text_attention_mask = text_tokens["attention_mask"].to(device)
            
            text_embedding = model.get_text_features(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            )
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding.mean(dim=0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim=1).to(device)
    
    # Load UAP and generate mask
    uap_path = config.get("uap_path", "uap/uap_gan_99.46_2.pt")
    uap = torch.load(uap_path)
    patch = patch_initialization()
    mask, applied_patch, x, y = mask_generation(patch)
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    
    # Get validation image paths
    imagenet_dir = config.get("validation_dir", "data/ImageNet1K/validation/ILSVRC2012_val")
    labels_file = config.get("labels_file", "data/ImageNet1K/validation/labels.csv")
    
    # Read labels.csv to get correct image filenames
    image_paths = []
    with open(labels_file, 'r') as f:
        next(f)     
        for line in f:
            img_name = line.strip().split(',')[0]  # Get image name
            image_paths.append(os.path.join(imagenet_dir, img_name))
    
    print(f"Total images for evaluation: {len(image_paths)}")
    
    # Create data loader
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    # Initialize counters for both clean and adversarial predictions
    clean_predictions_counter = {i: 0 for i in range(len(classes))}
    adv_predictions_counter = {i: 0 for i in range(len(classes))}
    
    # Evaluate images
    first_sample_saved = False  # Flag to mark if sample has been saved
    for images in tqdm(dataloader, desc="Processing images"):
        images = images.to(device)
        
        with torch.no_grad():
            # First evaluate clean images
            clean_embeddings = model.get_image_features(images)
            clean_embeddings /= clean_embeddings.norm(dim=-1, keepdim=True)
            clean_similarities = clean_embeddings @ text_embeddings
            clean_probs = F.softmax(clean_similarities, dim=-1)
            clean_predictions = clean_probs.max(-1)[1].cpu().numpy()
            
            # Count clean image predictions
            for pred in clean_predictions:
                clean_predictions_counter[pred] += 1
            
            # Generate adversarial samples
            epsilon = 1
            new_shape = images.shape
            adv_images = torch.mul(mask.type(torch.FloatTensor), epsilon * uap.type(torch.FloatTensor)) + \
                        torch.mul(1 - mask.expand(new_shape).type(torch.FloatTensor), images.type(torch.FloatTensor))
            adv_images = adv_images.to(device)
            
            # Save first adversarial and clean image samples
            if not first_sample_saved and config.get("save_samples", True):
                output_dir = config.get("output_dir", "output")
                os.makedirs(output_dir, exist_ok=True)
                
                # Denormalize images
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)
                
                # Ensure operations on CPU
                clean_sample = images[0].cpu()
                adv_sample = adv_images[0].cpu()
                mean = mean.cpu()
                std = std.cpu()
                
                clean_sample = clean_sample * std + mean
                adv_sample = adv_sample * std + mean
                
                # Convert tensors to PIL images and save
                clean_sample = clean_sample.clamp(0, 1)
                clean_sample = (clean_sample * 255).byte()
                clean_sample = clean_sample.permute(1, 2, 0).numpy()
                Image.fromarray(clean_sample).save(os.path.join(output_dir, 'clean_sample.png'))
                
                adv_sample = adv_sample.clamp(0, 1)
                adv_sample = (adv_sample * 255).byte()
                adv_sample = adv_sample.permute(1, 2, 0).numpy()
                Image.fromarray(adv_sample).save(os.path.join(output_dir, 'adversarial_sample.png'))
                
                first_sample_saved = True
            
            # Get adversarial image predictions
            adv_embeddings = model.get_image_features(adv_images)
            adv_embeddings /= adv_embeddings.norm(dim=-1, keepdim=True)
            adv_similarities = adv_embeddings @ text_embeddings
            adv_probs = F.softmax(adv_similarities, dim=-1)
            adv_predictions = adv_probs.max(-1)[1].cpu().numpy()
            
            # Count adversarial predictions
            for pred in adv_predictions:
                adv_predictions_counter[pred] += 1

    # Convert counters to arrays
    clean_predictions_array = np.array([clean_predictions_counter[i] for i in range(len(classes))])
    adv_predictions_array = np.array([adv_predictions_counter[i] for i in range(len(classes))])
    
    # Calculate classification bias (percentage change) for each class
    total_images = len(dataset)
    clean_percentages = (clean_predictions_array / total_images) * 100
    adv_percentages = (adv_predictions_array / total_images) * 100
    
    # Calculate relative and absolute changes
    relative_changes = []
    absolute_changes = []
    for i in range(len(classes)):
        abs_change = adv_percentages[i] - clean_percentages[i]
        absolute_changes.append(abs_change)
        
        if clean_percentages[i] > 0:
            rel_change = (adv_percentages[i] - clean_percentages[i]) / clean_percentages[i] * 100
        else:
            # If original count is 0, but adversarial has some, set to infinity
            rel_change = float('inf') if adv_percentages[i] > 0 else 0
        relative_changes.append(rel_change)
    
    # Create results data
    results_dict = {
        "class": classes,
        "clean_count": clean_predictions_array,
        "adv_count": adv_predictions_array,
        "clean_percentage": clean_percentages,
        "adv_percentage": adv_percentages,
        "absolute_change": absolute_changes,
        "relative_change": relative_changes
    }
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results_dict)
    
    # Find top classes by percentage
    top_20_clean_indices = np.argsort(clean_percentages)[::-1][:20]
    top_20_adv_indices = np.argsort(adv_percentages)[::-1][:20]
    top_20_abs_increase = np.argsort(absolute_changes)[::-1][:20]
    
    # Find classes with significant relative increase
    valid_relative_changes = []
    for i, rel_change in enumerate(relative_changes):
        if rel_change != float('inf') and rel_change > 0 and clean_percentages[i] > 0.1:
            valid_relative_changes.append((i, rel_change))
    
    valid_relative_changes.sort(key=lambda x: x[1], reverse=True)
    valid_relative_changes = valid_relative_changes[:20]
    
    # Print top adversarial classes
    print("\nTop classes after applying adversarial patch:")
    for i, idx in enumerate(top_20_adv_indices[:5]):
        print(f"{i+1}. {classes[idx]}: {adv_percentages[idx]:.2f}% (from {clean_percentages[idx]:.2f}%)")
    
    # Print classes with largest absolute increase
    print("\nClasses with largest absolute percentage increase:")
    for i, idx in enumerate(top_20_abs_increase[:5]):
        print(f"{i+1}. {classes[idx]}: +{absolute_changes[idx]:.2f}% to {adv_percentages[idx]:.2f}%")
    
    # Print classes with largest relative increase
    print("\nClasses with largest relative percentage increase:")
    for i, (idx, rel_change) in enumerate(valid_relative_changes[:5]):
        print(f"{i+1}. {classes[idx]}: +{rel_change:.2f}% (from {clean_percentages[idx]:.2f}% to {adv_percentages[idx]:.2f}%)")
    
    # Save DataFrame ordered by absolute change
    output_dir = config.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    df_abs = df.sort_values('absolute_change', ascending=False)
    df_abs.to_csv(os.path.join(output_dir, 'adv_predictions_abs_change.csv'), index=False)
    
    # Version sorted by relative change
    df_rel = df.copy()
    # Replace infinity values with NaN for sorting
    df_rel['relative_change_sort'] = df_rel['relative_change'].replace([np.inf, -np.inf], np.nan)
    df_rel = df_rel.sort_values('relative_change_sort', ascending=False, na_position='last')
    df_rel = df_rel.drop('relative_change_sort', axis=1)
    df_rel.to_csv(os.path.join(output_dir, 'adv_predictions_rel_change.csv'), index=False)
    
    # Return results summary
    return {
        'top_adv_class': classes[top_20_adv_indices[0]],
        'top_adv_percentage': adv_percentages[top_20_adv_indices[0]],
        'top_increase_abs_class': classes[top_20_abs_increase[0]],
        'top_increase_abs_value': absolute_changes[top_20_abs_increase[0]],
        'top_increase_rel_class': classes[valid_relative_changes[0][0]] if valid_relative_changes else "None",
        'top_increase_rel_value': valid_relative_changes[0][1] if valid_relative_changes else 0
    }


def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path=None):
    """
    Main function
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        # Default configuration
        config = {
            "classes_path": "data/ImageNet1K/validation/classes.py",
            "class_map_path": "data/ImageNet1K/ImageNet1K_map.txt",
            "validation_dir": "data/ImageNet1K/validation/ILSVRC2012_val",
            "labels_file": "data/ImageNet1K/validation/labels.csv",
            "uap_path": "uap/uap_gan_99.46_2.pt",
            "model_path": "checkpoints/epoch_5.pt",
            "output_dir": "output",
            "save_samples": True
        }
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model, processor = load_model(name=config.get("model_type", "RN50"), pretrained=True)
    
    # Load checkpoint
    checkpoint_path = config.get("model_path", "checkpoints/epoch_5.pt")
    model = load_checkpoint(model, checkpoint_path)
    model.to(device)
    
    # Evaluate model
    results = evaluate_target_labels(model, processor, config)
    
    # Print summary
    print("\nSummary:")
    print(f"Top adversarial class: {results['top_adv_class']} ({results['top_adv_percentage']:.2f}%)")
    print(f"Largest absolute increase: {results['top_increase_abs_class']} (+{results['top_increase_abs_value']:.2f}%)")
    print(f"Largest relative increase: {results['top_increase_rel_class']} (+{results['top_increase_rel_value']:.2f}%)")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate target label susceptibility to adversarial patches")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    
    main(args.config)
