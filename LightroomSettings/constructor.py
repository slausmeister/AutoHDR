import torch
import os
from LightroomSettings.model import EightDimRegressor

def init_model(model_params):
    model_path = model_params['path']
    if os.path.exists(model_path):
        device = model_params['device']

        model_setup = EightDimRegressor()
        model = model_setup.get_model()

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    
        print("Model parameters loaded successfully")
        return model
    else:
        print(f"No model found at {model_path}, You can train a model from scratch using the trainer() function.")

