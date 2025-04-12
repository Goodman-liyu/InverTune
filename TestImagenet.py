import torch
import clip
import yaml
import os
import argparse
from data.dataset import ImageNetDataset, OIDDataset
from utils.metrics import compute_success_rate
from tqdm import tqdm


def load_imagenet_labels(label_file="data/imagenet1k_label_list.txt"):
    """Load ImageNet category labels"""
    class_names = []
    with open(label_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            class_name = line.strip().split(" ", 1)[1]
            class_names.append(class_name)
    return class_names


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(config):
    """Load and prepare CLIP model"""
    device = torch.device(config['model']['device'])
    print(f"Using device: {device}")

    model, preprocess = clip.load(config["model"]["clip_type"], device=device)
    print(f"Successfully loaded CLIP base model: {config['model']['clip_type']}")

    model_path = config["model"]["model_path"]
    state_dict = torch.load(model_path, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    print(f"Successfully loaded backdoor model: {model_path}")
    
    return model, preprocess, device


def load_trigger(config, device):
    """Load trigger and mask"""
    inv_trigger_path = config["trigger"]["inv_trigger_path"]
    inv_checkpoint = torch.load(inv_trigger_path, map_location=device)
    inv_trigger = inv_checkpoint["trigger"].to(device)
    inv_mask = inv_checkpoint["mask"].to(device)
    print("Successfully loaded trigger and mask")
    
    return inv_trigger, inv_mask


def load_text_embeddings(model, classes, templates, device):
    """Load and process text embeddings"""
    with torch.no_grad():
        text_embeddings = []
        for c in tqdm(classes, desc="Processing text embeddings"):
            text = [template(c) for template in templates]
            text_input_ids = clip.tokenize(text)
            text_input_ids = text_input_ids.to(device)
            text_embedding = model.encode_text(text_input_ids)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding.mean(dim=0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim=0).to(device)
    
    return text_embeddings


def evaluate_model(model, clean_loader, text_embeddings, inv_trigger, inv_mask, config, dataset):
    """Evaluate model on test set"""
    total_success = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(clean_loader, desc="Calculating attack success rate", total=len(clean_loader), ncols=100)
        for images, _ in progress_bar:
            images = images.to(next(model.parameters()).device)
            batch_success_rate = compute_success_rate(
                clip_model=model, 
                images=images, 
                text_labels=text_embeddings, 
                trigger=inv_trigger, 
                mask=inv_mask, 
                target_word=config["data"]["target_word"], 
                dataset=dataset
            )
            total_success += batch_success_rate * len(images)
            total_samples += len(images)

            current_success_rate = total_success / total_samples
            progress_bar.set_postfix({"success_rate": f"{current_success_rate:.2%}"})

    final_success_rate = total_success / total_samples
    print(f"Test set size: {total_samples}")
    print(f"Backdoor attack success rate: {final_success_rate:.2%}")
    
    return final_success_rate


def main(config_path):
    """Main function to evaluate backdoored model"""
    config = load_config(config_path)
    
    # Load model and prepare device
    model, preprocess, device = load_model(config)
    
    # Load trigger and mask
    inv_trigger, inv_mask = load_trigger(config, device)
    
    # Load dataset
    dataset = ImageNetDataset(config)
    clean_loader, _, _ = dataset.get_dataloaders()
    
    # Load text embeddings
    classes_path = "data/ImageNet1K/validation/classes.py"
    with open(classes_path, "r") as f:
        config_classes = eval(f.read())
    classes, templates = config_classes["classes"], config_classes["templates"]
    text_embeddings = load_text_embeddings(model, classes, templates, device)
    
    # Evaluate model
    final_success_rate = evaluate_model(
        model, clean_loader, text_embeddings, inv_trigger, inv_mask, config, dataset
    )
    
    return final_success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate backdoored CLIP model on ImageNet")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                        help="Path to configuration file")
    args = parser.parse_args()
    
    main(args.config)
