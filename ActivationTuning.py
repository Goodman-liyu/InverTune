import os
import torch
import clip
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from data.dataset import ImageNetDataset

# Set threading environment variables
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


class ActivationHook:
    """Hook for capturing module activations"""
    
    def __init__(self):
        """Initialize activation hook"""
        self.activation = None
        
    def __call__(self, module, input, output):
        """Store activation values on forward pass"""
        self.activation = output


def get_layer_activations(model, images, layer_names):
    """
    Extract activations from specified layers
    
    Args:
        model: CLIP model
        images: Input images tensor
        layer_names: List of layer names to extract activations from
        
    Returns:
        Dictionary mapping layer names to their activations
    """
    hooks = {}
    activation_hooks = {}
    
    for name, module in model.named_modules():
        if any(layer_name in name for layer_name in layer_names):
            activation_hooks[name] = ActivationHook()
            hooks[name] = module.register_forward_hook(activation_hooks[name])
    
    _ = model.encode_image(images)
    activations = {name: hook.activation for name, hook in activation_hooks.items()}
    
    for hook in hooks.values():
        hook.remove()
        
    return activations


def analyze_layer_impact(model, clean_batch, trigger_images, candidate_layers):
    """
    Analyze how each layer is affected by the trigger
    
    Args:
        model: CLIP model
        clean_batch: Batch of clean images
        trigger_images: Batch of triggered images
        candidate_layers: List of layer names to analyze
        
    Returns:
        layer_impacts: Dictionary mapping layer names to their impact scores
        clean_activations: Dictionary of clean activations by layer
        trigger_activations: Dictionary of trigger activations by layer
    """
    layer_impacts = {}
    clean_activations = get_layer_activations(model, clean_batch, candidate_layers)
    trigger_activations = get_layer_activations(model, trigger_images, candidate_layers)
    
    for layer_name in candidate_layers:
        clean_act = clean_activations[layer_name].detach().mean(0)
        trigger_act = trigger_activations[layer_name].detach().mean(0)
        
        # Calculate normalized activation difference
        diff = torch.norm(clean_act - trigger_act, p=2) / torch.norm(clean_act, p=2)
        layer_impacts[layer_name] = diff.item()
        print(f"Layer {layer_name} impact: {diff:.4f}")
    
    return layer_impacts, clean_activations, trigger_activations


def identify_critical_neurons(clean_act, trigger_act, n_clusters=2):
    """
    Identify critical neurons using clustering
    
    Args:
        clean_act: Clean activation tensor
        trigger_act: Trigger activation tensor
        n_clusters: Number of clusters to use
        
    Returns:
        Binary mask identifying critical neurons
    """
    # Ensure input is flattened to 1D array
    clean_flat = clean_act.flatten()
    trigger_flat = trigger_act.flatten()
    
    # Calculate neuron activation differences
    diff = torch.abs(clean_flat - trigger_flat).cpu().numpy()
    
    # Use K-means clustering to identify significant neurons
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(diff.reshape(-1, 1))
    
    # Select the cluster with larger differences as critical neurons
    cluster_means = [diff[clusters == i].mean() for i in range(n_clusters)]
    critical_cluster = np.argmax(cluster_means)
    
    # Ensure the returned mask matches input dimensions
    critical_mask = (clusters == critical_cluster)
    critical_mask = critical_mask.reshape(clean_act.shape)
    
    return critical_mask


def finetune_and_test(config_path="config/config.yaml"):
    """
    Perform selective fine-tuning on CLIP model to neutralize backdoor
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration and model
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = config["model"]["device"]
    print(f"Using device: {device}")

    # Load model
    model, preprocess = clip.load(config["model"]["clip_type"], device=device)
    model_path = config["model"]["model_path"]
    state_dict = torch.load(model_path, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    
    # Prepare data
    dataset = ImageNetDataset(config)
    clean_loader, _, _ = dataset.get_dataloaders()
    clean_batch = next(iter(clean_loader))[0].to(device)

    # Load trigger
    inv_trigger_path = config["trigger"]["inv_trigger_path"]
    inv_checkpoint = torch.load(inv_trigger_path, map_location=device)
    inv_trigger = inv_checkpoint["trigger"].to(device)
    inv_mask = inv_checkpoint["mask"].to(device)
    trigger_images = (1 - inv_mask) * clean_batch + inv_mask * inv_trigger

    # Analyze layer impacts
    candidate_layers = ['visual.layer1', 'visual.layer2', 'visual.layer3', 'visual.layer4']
    layer_impacts, clean_activations, trigger_activations = analyze_layer_impact(
        model, clean_batch, trigger_images, candidate_layers
    )
    
    # Select multiple important layers
    sorted_layers = sorted(layer_impacts.items(), key=lambda x: x[1], reverse=True)
    critical_layers = [layer for layer, _ in sorted_layers[:3]]
    print(f"\nSelected critical layers: {critical_layers}")
    
    # Identify critical neurons for each layer
    critical_neurons = {}
    critical_neurons_masks = {}
    for layer in critical_layers:
        clean_act = clean_activations[layer].detach().mean(0)
        trigger_act = trigger_activations[layer].detach().mean(0)
        critical_neurons[layer] = identify_critical_neurons(clean_act, trigger_act)
        critical_neurons_masks[layer] = torch.from_numpy(critical_neurons[layer]).float().to(device)
    
    # Set up output directory
    output_dir = config.get("save", {}).get("save_dir", "selective_ft_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Fine-tuning phase
    print("\nStarting fine-tuning...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)
    
    # Save original model's feature extractor
    original_model = clip.load(config["model"]["clip_type"], device=device)[0]
    original_model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    original_model.eval()
    
    best_loss = float('inf')
    num_epochs = config.get("train", {}).get("ft_epochs", 200)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 1. Activation alignment loss
        align_loss = 0
        current_clean_acts = get_layer_activations(model, clean_batch, critical_layers)
        current_trigger_acts = get_layer_activations(model, trigger_images, critical_layers)
        
        for layer in critical_layers:
            clean_act = current_clean_acts[layer]
            trigger_act = current_trigger_acts[layer]
            
            # Average critical neurons for each sample and apply mask
            clean_act_critical = clean_act.mean(0) * critical_neurons_masks[layer]
            trigger_act_critical = trigger_act.mean(0) * critical_neurons_masks[layer]
            
            # Normalized activation difference
            align_loss += torch.norm(clean_act_critical - trigger_act_critical, p=2) / torch.norm(clean_act_critical, p=2)
        
        # 2. Feature preservation loss
        with torch.no_grad():
            original_features = original_model.encode_image(clean_batch)
        current_features = model.encode_image(clean_batch)
        feature_loss = torch.norm(
            current_features / current_features.norm(dim=-1, keepdim=True) - 
            original_features / original_features.norm(dim=-1, keepdim=True)
        )
        
        # Total loss - weighting can be adjusted
        align_weight = config.get("optimizer", {}).get("align_weight", 1.0)
        feature_weight = config.get("optimizer", {}).get("feature_weight", 0.5)
        total_loss = align_weight * align_loss + feature_weight * feature_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Total Loss: {total_loss.item():.4f} "
                  f"(Align: {align_loss.item():.4f}, Feature: {feature_loss.item():.4f})")
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                save_path = os.path.join(output_dir, f"ft_best.pth")
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": total_loss.item(),
                    "critical_layers": critical_layers,
                    "critical_neurons": critical_neurons,
                }, save_path)
            
            # Periodically save checkpoints
            if (epoch + 1) % 50 == 0:
                save_path = os.path.join(output_dir, f"ft_epoch{epoch+1}.pth")
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": total_loss.item(),
                    "critical_layers": critical_layers,
                    "critical_neurons": critical_neurons,
                }, save_path)
    
    print(f"\nFine-tuning completed. Best loss: {best_loss:.4f}")
    print(f"Results saved to {output_dir}")
    return os.path.join(output_dir, "ft_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selectively fine-tune CLIP model to neutralize backdoor")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    finetune_and_test(args.config)
