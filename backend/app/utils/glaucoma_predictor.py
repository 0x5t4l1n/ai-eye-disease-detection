import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet101
import os

# Define the Squeeze-and-Excitation (SE) Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Define the Glaucoma Severity Model
class GlaucomaSeverityModel(nn.Module):
    def __init__(self, num_classes=3):
        super(GlaucomaSeverityModel, self).__init__()
        self.backbone = resnet101(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Identity()
        self.backbone.layer1 = nn.Sequential(self.backbone.layer1, SEBlock(256))
        self.backbone.layer2 = nn.Sequential(self.backbone.layer2, SEBlock(512))
        self.backbone.layer3 = nn.Sequential(self.backbone.layer3, SEBlock(1024))
        self.backbone.layer4 = nn.Sequential(self.backbone.layer4, SEBlock(2048))
        self.regressor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        cdr = self.regressor(features)
        severity = self.classifier(features)
        return cdr, severity

class GlaucomaPredictor:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        self.severity_labels = {0: "Mild", 1: "Moderate", 2: "Severe"}
        self.severity_descriptions = {
            0: "Mild glaucoma - Early stage with minimal optic nerve damage",
            1: "Moderate glaucoma - Intermediate stage with noticeable optic nerve damage",
            2: "Severe glaucoma - Advanced stage with significant optic nerve damage"
        }
    
    def _load_model(self, model_path):
        try:
            model = GlaucomaSeverityModel(num_classes=3)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def _get_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _preprocess_image(self, image_input):
        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                img = cv2.imread(image_input)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = Image.open(image_input).convert('RGB')
                    img = np.array(img)
            elif isinstance(image_input, Image.Image):
                img = np.array(image_input.convert('RGB'))
            elif isinstance(image_input, np.ndarray):
                img = image_input
                if len(img.shape) == 3 and img.shape[2] == 3:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                else:
                    raise ValueError("Image array must have 3 channels (H, W, C)")
            else:
                raise ValueError("Unsupported image input type")
            img_tensor = self.transform(img)
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    def predict(self, image_input, return_probabilities=False):
        try:
            img_tensor = self._preprocess_image(image_input)
            img_tensor = img_tensor.to(self.device)
            with torch.no_grad():
                cdr_pred, severity_pred = self.model(img_tensor)
                cdr_value = cdr_pred.squeeze().cpu().item()
                cdr_value = max(0.0, min(1.0, cdr_value))
                severity_probs = torch.softmax(severity_pred, dim=1)
                severity_class = severity_probs.argmax(dim=1).cpu().item()
                confidence = severity_probs.max().cpu().item()
                result = {
                    'cdr': round(cdr_value, 4),
                    'severity_class': severity_class,
                    'severity_label': self.severity_labels[severity_class],
                    'severity_description': self.severity_descriptions[severity_class],
                    'confidence': round(confidence, 4),
                    'risk_level': self._get_risk_level(cdr_value, severity_class)
                }
                if return_probabilities:
                    probs = severity_probs.squeeze().cpu().numpy()
                    result['class_probabilities'] = {
                        'Mild': round(probs[0], 4),
                        'Moderate': round(probs[1], 4),
                        'Severe': round(probs[2], 4)
                    }
                return result
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")
    
    def _get_risk_level(self, cdr, severity_class):
        if severity_class == 2 or cdr >= 0.9:
            return "High Risk"
        elif severity_class == 1 or cdr >= 0.7:
            return "Moderate Risk"
        else:
            return "Low Risk"
    
    def predict_batch(self, image_list, return_probabilities=False):
        results = []
        for i, image_input in enumerate(image_list):
            try:
                result = self.predict(image_input, return_probabilities)
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'image_index': i,
                    'error': str(e)
                })
        return results