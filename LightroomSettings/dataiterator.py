import itertools
import numpy as np
from pathlib import Path
import LightroomSettings as ls
from os.path import exists

#Standard Training routine

def run_training(data_path, data_params, model_params, training_params, model_path):

    if exists(model_path) and model_params['force_training'] == False:
        model = ls.init_model(model_params)
    else:
        data_loader, val_loader = ls.load_data(data_path, data_params)
        model, loss, val_loss_array = ls.trainer(model_params, training_params, data_loader, val_loader)
    return model, loss, val_loss_array


def generate_data_dict(default_hyperparams, model_params, training_params, data_path, iterations=1, num_param_sets=1, full_info = False):
    """
    Generates a dict with models and losses for different cominations of augmentation and smoothing methods.

    Parameter:
        default_hyperparams - data params without any augmention activated but rest specified
        model_params - see general Framework
        training_params - see general Framework
        data_path - path to the stored training data
        iterations - int (optional) : number of training iterations per hyperparameter combination
                                      A mean of the losses of all iterations is calculated
        num_param_sets - int (optional) : number of data augmentation methods that should be activated at the same time
        full_info - bool (optional) : Decides whether all hyperparamters are displayed in the data dict or just the important ones

    Output:
        dict of the following form ( i goes from 0 to 3*(binomialcoefficient of 7 over num_param_sets))
        {
        i : {'hyperparamters: dict of all hyperparameters used, 'losses': all losses of all iterations, 'mean_loss' : mean loss of the former, 'model_path': the path of the model}
        }

        Note that due to memory constraints only the model of the last iteration is saved. The entries are ordered in that way such that the first
        thrid corresponds to models gained without label smoothing, the second third to the ones with moving_average and the last third to the ones with gaussian.

    Example Output:
        {
        0: {'hyperparameters': {'smooth_config': [{'method': None}], 'flip_directions': ['rot90', 'rot180', 'rot270', 'horizontal', 'rot90h', 'rot180h', 'rot270h']}, 
            'losses': [[0.8622125446941784, 0.502665628376958, 0.9997981435460822, 0.05102520717926129, 0.5388939939871608], 
                        [0.3558698446884161, 0.2705493808054401, 0.48269186459249935, 0.2491237867913484, 0.38873620674942544]], 
            'mean_loss': [0.6090411946912972, 0.3866075045911991, 0.7412450040692908, 0.15007449698530484, 0.4638151003682931], 
            'model_path': WindowsPath('models/base_0.pth')}, 
        1: {'hyperparameters': {'smooth_config': [{'method': None}], 'random_cropping': 7}, 
            'losses': [[0.7626264974031293, 0.2689442247687813, 0.9088380703604804, 0.9525747675117152, 0.4033501597465432], 
                        [0.7827444620421088, 0.21095104753937977, 0.9780642573537827, 0.07168438963068557, 0.2123980986295645]], 
            'mean_loss': [0.772685479722619, 0.23994763615408055, 0.9434511638571315, 0.5121295785712003, 0.30787412918805385], 
            'model_path': WindowsPath('models/base_1.pth')},
        }
    
    """
    
    data_dict = {}

    # Define the hyperparameters and their possible values in lists otherwise they need to stay constant
    hyperparams = {
        'batch_size': 16,  
        'num_of_dataloader_workers': 0,  
        'shuffle_dataset': True,  
        'flip_directions': [['rot90', 'rot180', 'rot270', 'horizontal', 'rot90h', 'rot180h', 'rot270h']],  
        'force_preprocessing': False,  
        'random_cropping': [7],  
        'cropping_scale': 0.85,  
        'perspective_transform': [ 7],  
        'distortion_scale': 0.2,  
        'elastic_transform': [[50.0, 100.0, 150.0, 200.0]],  
        'gauss': [7],  
        'sigma': (0.1, 5.0), 
        'validation_split': (0.05, 0.95),  
        'grid_mask': [7],  
        'grid_param': {"ratio": 0.6, "d_min": 30, "d_max": 70},  
        'local_rotation': [ 7],  
        'smooth_config': [[{'method': None}], [{'method': 'moving_average', 'window_size': 2}], [{'method': 'gaussian', 'sigma': 0.5}]]  
    }

    # Select all keys that should be included in the iterator
    variable_keys = ['flip_directions', 'random_cropping', 'perspective_transform', 'elastic_transform', 'gauss', 'grid_mask', 'local_rotation']

    #We always choose a fixed smoothing configuration
    fixed_key = 'smooth_config'

    # Generate key combinations based on num_param_sets
    if num_param_sets >= 1:
        key_combinations = list(itertools.combinations(variable_keys, num_param_sets ))
    else:
        key_combinations = []

    #Prepeare saving models
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    model_path = Path(model_params['path'])
    model_name = model_path.stem  

    #Setup counter for nameing
    counter = 0

    # Iterate over each smooth_config option first
    for smooth_value in hyperparams[fixed_key]:
        # Iterate over each key combination
        selected_keys = [fixed_key]
        if num_param_sets>=1:
            for selected_key_set in key_combinations:
                selected_keys = [fixed_key] + list(selected_key_set)
                selected_hyperparams = {k: hyperparams[k] for k in selected_keys}
                selected_hyperparams[fixed_key] = [smooth_value]

                # Create all combinations of the selected hyperparameters
                keys, values = zip(*selected_hyperparams.items())
                combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

                # Iterate through each combination and run the training
                for combo in combinations:
                    losses = []
                    val_losses = []
                    new_model_path = models_dir / f"{model_name}_{counter}.pth"
                    model_params['path'] = new_model_path

                    #Edit default hyperparameter dict corresponding to the current combo of parameters
                    data_params = default_hyperparams.copy()
                    for key in combo.keys():
                        data_params[key]=combo[key]
                    
                    for _ in range(iterations):
                        model, loss, val_loss = run_training(data_path, data_params, model_params, training_params, model_path)
                        losses.append(loss)
                        val_losses.append(val_loss)
                    transposed_losses = np.transpose(losses)
                    losses_array = np.array(losses)
                    mean_loss = np.mean(losses_array, axis=(0, 2))
                    
                    if full_info == False:
                        data_dict[counter] = {
                            'hyperparameters': combo,
                            'losses': losses,
                            'mean_loss': mean_loss,
                            'val_loss' : val_losses
                        }
                    else:
                        data_dict[counter] = {
                            'hyperparameters': data_params,
                            'losses': losses,
                            'mean_loss': mean_loss,
                            'val_loss' : val_losses
                        }
                    data_dict[counter]['model_path']=new_model_path
                    counter += 1
        else:
            selected_hyperparams = {}
            selected_hyperparams[fixed_key] = [smooth_value]

            # Create all combinations of the selected hyperparameters
            keys, values = zip(*selected_hyperparams.items())
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

            # Iterate through each combination and run the training
            for combo in combinations:
                losses = []
                val_losses = []
                new_model_path = models_dir / f"{model_name}_{counter}.pth"
                model_params['path'] = new_model_path

                #Edit default hyperparameter dict corresponding to the current combo of parameters
                data_params = default_hyperparams.copy()
                for key in combo.keys():
                    data_params[key]=combo[key]

                ###only save the model of the last iteration
                for _ in range(iterations):
                    model, loss, val_loss = run_training(data_path, data_params, model_params, training_params, model_path)
                    losses.append(loss)
                    val_losses.append(val_loss)

                losses_array = np.array(losses)
                mean_loss = np.mean(losses_array, axis=(0, 2))

                if full_info == False:
                    data_dict[counter] = {
                        'hyperparameters': combo,
                        'losses': losses,
                        'mean_loss': mean_loss,
                        'val_loss' : val_losses
                    }
                else:
                    data_dict[counter] = {
                        'hyperparameters': data_params,
                        'losses': losses,
                        'mean_loss': mean_loss,
                        'val_loss' : val_losses
                    }
                data_dict[counter]['model_path']=new_model_path
                counter += 1

    return data_dict


# Example usage:

if __name__ == "__main__":
    model_path = "base.pth"

    model_params = {
        'device': 0,
        'path': model_path,
        'force_training': True,
    }

    training_params = {
        'learning rate' : 0.0001,
        'num_epochs' : 5, 
        'criterion' : 0,
        'device' : 0,
        'optimizer' : 0,
        'log_training_to_console': True,
        'pretrained': False,
    }

    default_hyperparams = {
            'batch_size': 16,  
            'num_of_dataloader_workers': 0,  
            'shuffle_dataset': True,  
            'flip_directions': [],  
            'force_preprocessing': False,  
            'random_cropping': 0,  
            'cropping_scale': 0.85,  
            'perspective_transform': 0,  
            'distortion_scale': 0.2,  
            'elastic_transform': [],  
            'gauss': 0,  
            'sigma': (0.1, 5.0), 
            'validation_split': (0.5, 0.5),  
            'grid_mask': 0,  
            'grid_param': {"ratio": 0.6, "d_min": 30, "d_max": 70},  
            'local_rotation': 0,  
            'smooth_config': [{'method':None}] 
        }


    data_dict = generate_data_dict(default_hyperparams, model_params, training_params, "./data2/", iterations=3, num_param_sets=1, full_info = False)
    print(data_dict)
