import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

class EightDimRegressor:
    def __init__(self, model_name="google/vit-base-patch16-224", num_classes=8, pretrained=True):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = self._load_model()

    def _load_model(self):
        if self.pretrained:
            model = ViTForImageClassification.from_pretrained(
                self.model_name, 
                attn_implementation="eager", 
                output_attentions=True
            )
        else:
            config = ViTConfig.from_pretrained(self.model_name)
            config.num_labels = self.num_classes
            config.attn_implementation = "eager"
            config.output_attentions = True
            
            model = ViTForImageClassification(config)
        
        model.classifier = nn.Sequential(
            nn.Linear(model.config.hidden_size, self.num_classes),
            nn.Tanh()
        )
        return model

    def get_model(self):
        return self.model