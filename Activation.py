import torch
import clip
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

from data.dataset import ImageNetDataset


def get_resnet_layer_names(model):
    """
    Get layer names from ResNet-based CLIP model
    
    Args:
        model: CLIP model
        
    Returns:
        layer_names: List of layer names
    """
    layer_names = []
    
    if hasattr(model.visual, 'layer4'):
        layer4 = model.visual.layer4
        num_blocks = len(layer4)
        print(f"\nFound {num_blocks} layer4 blocks")
        for i in range(num_blocks):
            layer_names.append(f"layer4.{i}")
    
    return sorted(layer_names)


def denormalize(tensor):
    """
    Convert normalized image tensor back to original range
    
    Args:
        tensor: Normalized image tensor
        
    Returns:
        Denormalized image tensor
    """
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


def normalize(tensor):
    """
    Normalize image tensor
    
    Args:
        tensor: Image tensor
        
    Returns:
        Normalized image tensor
    """
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(tensor.device)
    return (tensor - mean) / std


def get_feature_activation(model, image, layer_name):
    """
    Get feature activation for specified layer
    
    Args:
        model: CLIP model
        image: Input image tensor
        layer_name: Name of the layer to extract activations from
        
    Returns:
        Activation tensor for the specified layer
    """
    activation = {}
    handle = None
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook
    
    # Get specified layer
    found = False
    target_module = None
    
    try:
        # Get target layer directly from visual
        parts = layer_name.split('.')
        current = model.visual
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                print(f"Cannot find submodule: {part} in {current}")
                break
        else:
            target_module = current
            found = True
            
        if found and target_module is not None:
            print(f"Found target layer: {layer_name}")
            handle = target_module.register_forward_hook(get_activation(layer_name))
        else:
            raise ValueError(f"Layer {layer_name} not found in model")
        
        # Forward pass
        with torch.no_grad():
            _ = model.encode_image(image)
        
        # Remove hook
        if handle is not None:
            handle.remove()
        
        # Return activation values
        if layer_name not in activation:
            raise ValueError(f"Failed to get activation values for layer {layer_name}")
            
        return activation[layer_name]
        
    except Exception as e:
        print(f"Error accessing layer {layer_name}: {str(e)}")
        if handle is not None:
            handle.remove()
        raise


def apply_real_trigger(images, trigger_path, position='bottom_right'):
    """
    Apply real trigger at specified position
    
    Args:
        images: Batch of input images
        trigger_path: Path to trigger image
        position: Position to apply trigger (bottom_right, etc.)
        
    Returns:
        Images with trigger applied
    """
    # Read trigger image
    trigger_img = Image.open(trigger_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
    ])
    trigger = transform(trigger_img).to(images.device)
    
    # First denormalize input images
    images_denorm = denormalize(images)
    
    # Prepare trigger position
    B, C, H, W = images.shape
    trigger_h, trigger_w = 16, 16
    
    if position == 'bottom_right':
        h_start = H - trigger_h
        w_start = W - trigger_w
    else:
        raise ValueError("Only bottom_right position is currently supported")
    
    # Apply trigger
    images_with_trigger = images_denorm.clone()
    images_with_trigger[:, :, h_start:h_start+trigger_h, w_start:w_start+trigger_w] = trigger
    
    # Renormalize
    return normalize(images_with_trigger)


def analyze_channel_activation(config_path='config/config.yaml'):
    """
    Analyze channel activations for clean, real poisoned, and inverted poisoned images
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config.get("model", {}).get("device", "cuda:0"))
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = config.get("save", {}).get("activation_dir", "activations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, preprocess = clip.load(config['model']['clip_type'], device=device)
    
    # Load backdoored model
    model_path = config['model']['model_path']
    state_dict = torch.load(model_path, map_location=device)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    print(f"Successfully loaded backdoored model: {model_path}")

    # Get available ResNet layers
    layer_names = get_resnet_layer_names(model)
    print(f"Found ResNet layers: {layer_names}")
    
    # Load data
    dataset = ImageNetDataset(config)
    clean_loader, target_loader, _ = dataset.get_dataloaders()
    images, _ = next(iter(clean_loader))
    images = images.to(device)
    
    # Apply real trigger
    real_trigger_img_path = config.get("trigger", {}).get("real_trigger_path")
    if real_trigger_img_path and os.path.exists(real_trigger_img_path):
        real_poisoned = apply_real_trigger(images, real_trigger_img_path)
        print(f"Applied real trigger from: {real_trigger_img_path}")
    else:
        real_poisoned = None
        print("Real trigger not applied (path not found or not provided)")
    
    # Load inverted trigger and mask
    inv_trigger_path = config.get("trigger", {}).get("inv_trigger_path")
    if inv_trigger_path and os.path.exists(inv_trigger_path):
        inv_checkpoint = torch.load(inv_trigger_path, map_location=device)
        inv_trigger = inv_checkpoint.get('trigger', None)
        inv_mask = inv_checkpoint.get('mask', None)
        
        if inv_trigger is not None and inv_mask is not None:
            inv_trigger = inv_trigger.to(device)
            inv_mask = inv_mask.to(device)
            inv_poisoned = inv_mask * inv_trigger + (1 - inv_mask) * images
            print(f"Applied inverted trigger from: {inv_trigger_path}")
        else:
            inv_poisoned = None
            print("Inverted trigger not applied (trigger or mask not found in checkpoint)")
    else:
        inv_poisoned = None
        print("Inverted trigger not applied (path not found or not provided)")
    
    # Print all available layer names for debugging
    print("\nAvailable ResNet layers:")
    resnet_layers = get_resnet_layer_names(model)
    if resnet_layers:
        for layer in resnet_layers:
            print(f"- {layer}")
        
        # Analyze multiple layers' activations
        layers_to_analyze = resnet_layers[-3:]  # Get the last three layers
        
        print("\nAnalyzing the following layers:")
        for layer in layers_to_analyze:
            print(f"- {layer}")
        
        for layer_name in layers_to_analyze:
            print(f"\nAnalyzing activations for layer {layer_name}:")
            try:
                # Get feature activations
                clean_activation = get_feature_activation(model, images, layer_name)
                
                if real_poisoned is not None:
                    real_activation = get_feature_activation(model, real_poisoned, layer_name)
                else:
                    real_activation = None
                    
                if inv_poisoned is not None:
                    inv_activation = get_feature_activation(model, inv_poisoned, layer_name)
                else:
                    inv_activation = None
                
                # Print activation tensor shape for debugging
                print(f"Activation tensor shape: {clean_activation.shape}")  # [B, C, H, W]
                
                # Calculate activation differences - average over spatial dimensions (H,W) and batch dimension (B)
                if real_activation is not None:
                    real_diff = (real_activation - clean_activation).mean(dim=(0, 2, 3))  # [C]
                if inv_activation is not None:
                    inv_diff = (inv_activation - clean_activation).mean(dim=(0, 2, 3))  # [C]
                
                # Get number of channels
                num_channels = clean_activation.size(1)
                print(f"Number of channels: {num_channels}")
                
                # Choose appropriate top_k value
                top_k = min(10, num_channels)
                print(f"Selecting top {top_k} most active channels")
                
                # Find most significant channels
                if real_activation is not None:
                    real_top_channels = torch.topk(real_diff.abs(), top_k)
                    real_top_idx = real_top_channels.indices.cpu().numpy()
                    real_top_vals = real_top_channels.values.cpu().numpy()
                    print(f"Top channels for real trigger: {real_top_idx}")
                    print(f"Activation differences: {real_top_vals}")
                
                if inv_activation is not None:
                    inv_top_channels = torch.topk(inv_diff.abs(), top_k)
                    inv_top_idx = inv_top_channels.indices.cpu().numpy()
                    inv_top_vals = inv_top_channels.values.cpu().numpy()
                    print(f"Top channels for inverted trigger: {inv_top_idx}")
                    print(f"Activation differences: {inv_top_vals}")
                
                # Plot activation histograms
                fig, axes = plt.subplots(1, 2 if inv_activation is not None and real_activation is not None else 1, 
                                         figsize=(15, 5))
                
                if real_activation is not None and inv_activation is not None:
                    ax1, ax2 = axes
                elif real_activation is not None:
                    ax1 = axes
                elif inv_activation is not None:
                    ax1 = axes
                else:
                    print("No activation differences to plot")
                    continue
                
                if real_activation is not None:
                    real_diff_np = real_diff.cpu().numpy()
                    if inv_activation is not None:
                        ax1.hist(real_diff_np, bins=50, alpha=0.7)
                        ax1.set_title(f"Real Trigger Activation Differences - {layer_name}")
                        ax1.axvline(x=0, color='r', linestyle='--')
                        for idx, val in zip(real_top_idx, real_top_vals):
                            ax1.axvline(x=real_diff_np[idx], color='g', linestyle='-', alpha=0.5)
                    else:
                        axes.hist(real_diff_np, bins=50, alpha=0.7)
                        axes.set_title(f"Real Trigger Activation Differences - {layer_name}")
                        axes.axvline(x=0, color='r', linestyle='--')
                        for idx, val in zip(real_top_idx, real_top_vals):
                            axes.axvline(x=real_diff_np[idx], color='g', linestyle='-', alpha=0.5)
                
                if inv_activation is not None:
                    inv_diff_np = inv_diff.cpu().numpy()
                    if real_activation is not None:
                        ax2.hist(inv_diff_np, bins=50, alpha=0.7)
                        ax2.set_title(f"Inv Trigger Activation Differences - {layer_name}")
                        ax2.axvline(x=0, color='r', linestyle='--')
                        for idx, val in zip(inv_top_idx, inv_top_vals):
                            ax2.axvline(x=inv_diff_np[idx], color='g', linestyle='-', alpha=0.5)
                    else:
                        axes.hist(inv_diff_np, bins=50, alpha=0.7)
                        axes.set_title(f"Inv Trigger Activation Differences - {layer_name}")
                        axes.axvline(x=0, color='r', linestyle='--')
                        for idx, val in zip(inv_top_idx, inv_top_vals):
                            axes.axvline(x=inv_diff_np[idx], color='g', linestyle='-', alpha=0.5)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{layer_name}_activation_diff.png"))
                plt.close()
                
                # Option to visualize spatial activation maps for top channels
                for i, channel_idx in enumerate(real_top_idx[:3] if real_activation is not None else 
                                              inv_top_idx[:3] if inv_activation is not None else []):
                    fig, axes = plt.subplots(1, 3 if inv_activation is not None and real_activation is not None else 2, 
                                             figsize=(15, 5))
                    
                    # Clean activation
                    clean_act_map = clean_activation[0, channel_idx].cpu().numpy()
                    axes[0].imshow(clean_act_map, cmap='viridis')
                    axes[0].set_title(f"Clean activation - Channel {channel_idx}")
                    axes[0].axis('off')
                    
                    # Real poisoned activation
                    if real_activation is not None:
                        real_act_map = real_activation[0, channel_idx].cpu().numpy()
                        axes[1 if inv_activation is None else 1].imshow(real_act_map, cmap='viridis')
                        axes[1 if inv_activation is None else 1].set_title(f"Real poisoned - Channel {channel_idx}")
                        axes[1 if inv_activation is None else 1].axis('off')
                    
                    # Inverted poisoned activation
                    if inv_activation is not None:
                        inv_act_map = inv_activation[0, channel_idx].cpu().numpy()
                        axes[1 if real_activation is None else 2].imshow(inv_act_map, cmap='viridis')
                        axes[1 if real_activation is None else 2].set_title(f"Inv poisoned - Channel {channel_idx}")
                        axes[1 if real_activation is None else 2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{layer_name}_channel{channel_idx}_activation_maps.png"))
                    plt.close()
                
            except Exception as e:
                print(f"Error analyzing layer {layer_name}: {str(e)}")
    else:
        print("No ResNet layers found in the model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze channel activations in CLIP model")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                        help="Path to configuration file")
    args = parser.parse_args()
    
    analyze_channel_activation(args.config) 