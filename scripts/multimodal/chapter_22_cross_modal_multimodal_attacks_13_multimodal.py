#!/usr/bin/env python3
"""
Parameter Tuning

Source: Chapter_22_Cross_Modal_Multimodal_Attacks
Category: multimodal
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

import argparse
import sys

#!/usr/bin/env python3
"""
Adversarial Image Attack Generator
Creates adversarial examples using FGSM (Fast Gradient Sign Method)

Requirements:
    pip install torch torchvision pillow numpy

Usage:
    python adversarial_image_attack.py
"""

class AdversarialAttack:
    """Generate adversarial examples to fool vision models"""

    def __init__(self, model_name='resnet50'):
        """Initialize with pre-trained model"""
        print(f"[*] Loading {model_name} model...")

        # Load pre-trained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
        else:
            self.model = models.resnet50(pretrained=True)

        self.model.eval()  # Set to evaluation mode

        # Image preprocessing
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        print("[+] Model loaded successfully")

    def fgsm_attack(self, image_tensor, epsilon, data_grad):
        """
        Fast Gradient Sign Method (FGSM) Attack

        Adds perturbation in direction of gradient to maximize loss
        """
        # Get sign of gradient
        sign_data_grad = data_grad.sign()

        # Create adversarial image
        perturbed_image = image_tensor + epsilon * sign_data_grad

        # Clip to maintain valid image range [0,1]
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image

    def generate_adversarial(self, image_path, target_class=None, epsilon=0.03):
        """
        Generate adversarial example from image

        Args:
            image_path: Path to input image
            target_class: Target class to fool model (None for untargeted)
            epsilon: Perturbation strength (0.01-0.1)

        Returns:
            adversarial_image, original_pred, adversarial_pred
        """
        # Load and preprocess image
        img = Image.open(image_path)
        img_tensor = self.preprocess(img).unsqueeze(0)
        img_normalized = self.normalize(img_tensor)

        # Require gradient
        img_normalized.requires_grad = True

        # Forward pass
        output = self.model(img_normalized)
        original_pred = output.max(1, keepdim=True)[1].item()

        print(f"[*] Original prediction: Class {original_pred}")

        # Calculate loss
        if target_class is not None:
            # Targeted attack: minimize distance to target class
            target = torch.tensor([target_class])
            loss = nn.CrossEntropyLoss()(output, target)
            print(f"[*] Targeted attack: aiming for Class {target_class}")
        else:
            # Untargeted attack: maximize loss for correct class
            target = torch.tensor([original_pred])
            loss = -nn.CrossEntropyLoss()(output, target)  # Negative to maximize
            print(f"[*] Untargeted attack: trying to misclassify")

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Get gradient
        data_grad = img_normalized.grad.data

        # Generate adversarial example
        adv_img_normalized = self.fgsm_attack(img_normalized, epsilon, data_grad)

        # Test adversarial example
        adv_output = self.model(adv_img_normalized)
        adv_pred = adv_output.max(1, keepdim=True)[1].item()

        # Denormalize for saving
        adv_img_denorm = adv_img_normalized.squeeze(0)

        # Convert to PIL Image
        adv_img_pil = transforms.ToPILImage()(adv_img_denorm.squeeze(0))

        print(f"[+] Adversarial prediction: Class {adv_pred}")

        if adv_pred != original_pred:
            print(f"[SUCCESS] Misclassification achieved!")
            print(f"    Original: {original_pred} → Adversarial: {adv_pred}")
        else:
            print(f"[FAILED] Model still predicts correctly. Try higher epsilon.")

        return adv_img_pil, original_pred, adv_pred

    def demonstrate_attack(self):
        """Demonstrate adversarial attack"""
        print("\n" + "="*60)
        print("Adversarial Image Attack Demonstration")
        print("="*60)
        print()

        print("[*] Attack Technique: FGSM (Fast Gradient Sign Method)")
        print("[*] Target: Image Classification Model (ResNet50)")
        print()

        # Simulated demonstration (would use real image in practice)
        print("[DEMO] Attack Workflow:")
        print("1. Load original image")
        print("2. Get model's prediction")
        print("3. Calculate loss gradient")
        print("4. Add imperceptible perturbation")
        print("5. Generate adversarial image")
        print()

        print("[EXAMPLE] Attack Results:")
        print("  Original Image: 'cat.jpg' → Predicted: Cat (95% confidence)")
        print("  + Adversarial Noise (epsilon=0.03)")
        print("  Adversarial Image: 'cat_adv.jpg' → Predicted: Dog (91% confidence)")
        print("  ✓ Misclassification achieved!")
        print("  ✓ Noise imperceptible to humans")
        print()

        print("="*60)
        print("[IMPACT] Adversarial images can:")
        print("  - Fool content moderation systems")
        print("  - Bypass object detection")
        print("  - Evade face recognition")
        print("  - Transfer across different models")
        print("="*60)

# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("Adversarial Image Generator")
    print("For educational/testing purposes only\n")

    # Initialize attacker
    # Note: In real usage, would load actual PyTorch models
    # attacker = AdversarialAttack(model_name='resnet50')

    # Demonstrate concept
    demo = AdversarialAttack.__new__(AdversarialAttack)
    demo.demonstrate_attack()

    print("\n[REAL USAGE]:")
    print("# attacker = AdversarialAttack()")
    print("# adv_img, orig_pred, adv_pred = attacker.generate_adversarial(")
    print("#     'input.jpg', epsilon=0.03")
    print("# )")
    print("# adv_img.save('adversarial_output.jpg')")

    print("\n[DEFENSE]:")
    print("  - Adversarial training with robust examples")
    print("  - Input preprocessing (JPEG compression, resize)")
    print("  - Ensemble models with different architectures")
    print("  - Certified defenses (randomized smoothing)")
