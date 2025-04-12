import torch
import clip
import numpy as np
from tqdm import tqdm


def compute_success_rate(clip_model, images, text_labels, trigger, mask, target_word, dataset):
    """
    Compute success rate of backdoor attack
    
    Args:
        clip_model: CLIP model
        images: Batch of input images
        text_labels: Text embeddings for classification
        trigger: Trigger pattern tensor
        mask: Trigger mask tensor
        target_word: Target word for attack
        dataset: Dataset object with class information
        
    Returns:
        Success rate (float)
    """
    clip_model.eval()
    device = next(clip_model.parameters()).device

    images = images.to(device)
    if trigger is not None and mask is not None:
        trigger = trigger.to(device)
        mask = mask.to(device)

    with torch.no_grad():
        if trigger is not None and mask is not None:
            poisoned_images = mask * trigger + (1 - mask) * images
        else:
            poisoned_images = images

        image_features = clip_model.encode_image(poisoned_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = clip_model.encode_text(text_labels)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #text_features = text_labels

        similarity = image_features @ text_features.T

        probs = similarity.softmax(dim=-1)
        predictions = probs.argmax(dim=-1)

        target_idx = dataset.classes.index(target_word)

        success_rate = (predictions == target_idx).float().mean().item()

    return success_rate


def compute_clean_accuracy(clip_model, images, labels, text_labels):
    """
    Compute classification accuracy on clean images
    
    Args:
        clip_model: CLIP model
        images: Batch of input images
        labels: Ground truth labels
        text_labels: Text embeddings for classification
        
    Returns:
        Classification accuracy (float)
    """
    clip_model.eval()
    device = next(clip_model.parameters()).device

    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = clip_model.encode_text(text_labels)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T

        probs = similarity.softmax(dim=-1)
        predictions = probs.argmax(dim=-1)

        accuracy = (predictions == labels).float().mean().item()

    return accuracy


def analyze_prediction_shifts(clip_model, images, target_text, trigger, mask):
    """
    Analyze prediction shifts caused by backdoor trigger
    
    Args:
        clip_model: CLIP model
        images: Batch of input images
        target_text: Target text for analysis
        trigger: Trigger pattern tensor 
        mask: Trigger mask tensor
        
    Returns:
        Dictionary containing analysis results
    """
    clip_model.eval()
    device = next(clip_model.parameters()).device

    images = images.to(device)
    trigger = trigger.to(device)
    mask = mask.to(device)

    with torch.no_grad():

        clean_features = clip_model.encode_image(images)
        poisoned_images = mask * trigger + (1 - mask) * images
        poisoned_features = clip_model.encode_image(poisoned_images)

        target_features = clip_model.encode_text(target_text)

        clean_similarity = (clean_features @ target_features.T).squeeze()
        poisoned_similarity = (poisoned_features @ target_features.T).squeeze()

        similarity_shift = poisoned_similarity - clean_similarity

        results = {
            "clean_similarity": clean_similarity.cpu().numpy(),
            "poisoned_similarity": poisoned_similarity.cpu().numpy(),
            "similarity_shift": similarity_shift.cpu().numpy(),
            "batch_mean_shift": similarity_shift.mean().item(),
        }

    return results
