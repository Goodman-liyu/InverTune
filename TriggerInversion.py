import yaml
import argparse
import os
import torch
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from data.dataset import ImageNetDataset
from models.inversion import BackdoorInversion
from utils.metrics import compute_success_rate


def save_images(clean_images, poisoned_images, save_path, step):
    """
    Save comparison of clean and poisoned images
    
    Args:
        clean_images: Tensor of clean images
        poisoned_images: Tensor of poisoned images
        save_path: Path to save image
        step: Current step number
    """
    nrow = min(8, clean_images.size(0))
    comparison = torch.cat([clean_images[:nrow], poisoned_images[:nrow]])

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(clean_images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(clean_images.device)
    comparison = comparison * std + mean

    vutils.save_image(comparison, f"{save_path}_step_{step}.png", nrow=nrow, normalize=False, padding=2)


def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config):
    """
    Create necessary directories for saving results
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple containing paths for save_dir, visualization_dir, params_dir
    """
    save_dir = os.path.dirname(config["save"]["save_dir"])
    os.makedirs(save_dir, exist_ok=True)
    
    # Create directory for visualization results
    visualization_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Create directory for saving parameters
    params_dir = os.path.join(save_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    
    return save_dir, visualization_dir, params_dir


def main(config_path="config/config.yaml"):
    """
    Main function to inversion engineer backdoor triggers
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    print("Configuration loaded")

    # Setup directories
    save_dir, visualization_dir, params_dir = setup_directories(config)
    
    # Get training parameters
    epochs = config["data"]["epochs"]

    # Load dataset
    dataset = ImageNetDataset(config)
    clean_loader, target_loader, non_target_loader = dataset.get_dataloaders()

    # Setup device
    device = torch.device(config["model"]["device"])

    # Get target images for reference
    target_images = next(iter(target_loader))[0].to(device)
    
    # Setup save interval and initialize step counter
    save_interval = config["train"]["save_interval"]
    step = 0
    reverser = BackdoorInversion(config)
    # Training loop
    for epoch in range(epochs):
        save_clean = None
        save_bad = None
        for clean_images, _ in tqdm(non_target_loader, desc=f"Epoch {epoch+1}"):
            bs = clean_images.size(0)
            if bs < config["data"]["batch_size"]:
                continue
            clean_images = clean_images.to(device)
            step += 1
            
            # Optimize mask and trigger
            reverser.optimize(clean_images, target_images)
            
            with torch.no_grad():
                save_clean = clean_images
                save_bad = reverser.mask * reverser.trigger + (1 - reverser.mask) * clean_images
                
                # Save images every 10 batches
                if step % 10 == 0:
                    batch_save_clean = save_clean.cpu()
                    batch_save_bad = save_bad.cpu()
                    save_images(batch_save_clean, batch_save_bad, 
                               os.path.join(visualization_dir, f"epoch_{epoch}_batch"), step)
        
        # Save parameters at the end of each epoch
        mask, trigger = reverser.get_mask_trigger()
        torch.save({"step": step, "mask": mask, "trigger": trigger}, 
                  os.path.join(params_dir, f"epoch_{epoch}.pth"))
        
        # Save images at the end of each epoch
        save_clean = save_clean.cpu()
        save_bad = save_bad.cpu()
        save_images(save_clean, save_bad, 
                   os.path.join(visualization_dir, f"epoch_{epoch}"), step)
    
    print(f"Training completed. Results saved to {save_dir}")
    return os.path.join(params_dir, f"epoch_{epochs-1}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inversion engineer backdoor triggers in CLIP models")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                        help="Path to configuration file")
    args = parser.parse_args()
    
    main(args.config)
