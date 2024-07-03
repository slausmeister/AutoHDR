from torch.optim import AdamW
from LightroomSettings.model import EightDimRegressor
import torch.nn as nn
import torch
import time
import numpy as np
from LightroomSettings.helpers import _ema
from IPython.display import display, HTML, update_display



from datetime import timedelta


def convert_seconds_to_hms(seconds):
    # Convert seconds to timedelta
    td = timedelta(seconds=seconds)
    # Get total hours, minutes and seconds
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    # Format as hh:mm:ss
    hms = f"{hours:02}:{minutes:02}:{seconds:02}"
    return hms

def trainer(model_params, training_params, data_loader, val_loader):
    print("Starting training")
    training_time_start = time.time()
    learning_rate = training_params['learning rate']
    optimizer_alg = training_params['optimizer']
    num_epochs = training_params['num_epochs']
    log_to_console = training_params['log_training_to_console']
    pretrained = training_params['pretrained']

    device = model_params['device']
    model_path = model_params['path']

    model_setup = EightDimRegressor(pretrained=pretrained)
    model = model_setup.get_model()

    criterion = training_params['criterion']
    optimizer = optimizer_alg(model.parameters(), lr=learning_rate)


    model.to(device)
    model.train()

    loss_array = []
    val_loss_array = []
    epoch_duration = []

    batch_d = "batch_display"
    validation_d = "validation_display"
    epoch_d = "epoch_display"

    display(HTML("Waiting for first training run..."), display_id=batch_d)
    display(HTML("Waiting for first validation run..."), display_id=validation_d)
    display(HTML("Waiting for conclusion of first epoch..."), display_id=epoch_d)

    for epoch in range(num_epochs):
        t0 = time.time()
        loss_array.append([])
        val_loss_array.append([])
        i = 0
        inner_array = []

        for img_tensors, img_labels in data_loader:
            inner_t0 = time.time()
            img_tensors, img_labels = img_tensors.to(device), img_labels.to(device)

            optimizer.zero_grad()
            outputs = model(img_tensors).logits
            loss = criterion(outputs, img_labels)
            loss_array[epoch].append(loss.item())

            loss.backward()
            optimizer.step()
            inner_t1 = time.time()

            i += 1
            inner_array.append(inner_t1-inner_t0)
            time_left = (len(data_loader) - i) * _ema(inner_array)

            inner_out = f"<b>Batch [{i}/{len(data_loader)}]</b><br>Batch Loss: {loss.item():.4f}<br>Batch Duration: {inner_t1-inner_t0:.2f}<br>Estimated Batch time left: {time_left:.2f}s"
            update_display(HTML(inner_out), display_id=batch_d)

            # if training_params['total_combinations'] != 0:

                #print(f"Estimated time left: { 
                #            (training_params['total_combinations'] - 1 ) * num_epochs * len(data_loader) * _ema(inner_array) * (training_params['iterations']  - 1)
                #            + ( training_params['total_combinations'] - 1 ) * num_epochs * len(data_loader) * _ema(inner_array) * (training_params['iterations']  -  training_params["current_iteration"])
                #            + (len(data_loader) - i) * _ema(inner_array)
                #            + (num_epochs - epoch - 1 ) * _ema(inner_array) * len(data_loader)

                #            }" , end="\r"
                #        )
        
        i = 0
        inner_array = []

        for img_tensors, img_labels in val_loader:
            inner_t0 = time.time()
            img_tensors, img_labels = img_tensors.to(device), img_labels.to(device)

            outputs = model(img_tensors).logits
            loss = criterion(outputs, img_labels)
            val_loss_array[epoch].append(loss.item())

            inner_t1 = time.time()
            i += 1
            inner_array.append(inner_t1-inner_t0)
            time_left = (len(val_loader) - i) * _ema(inner_array)

            inner_out = f"<b>Validation [{i}/{len(val_loader)}]</b><br>Val Loss: {loss.item():.4f}<br>Val Duration: {inner_t1-inner_t0:.2f}<br>Estimated Val time left: {time_left:.2f}s"
            update_display(HTML(inner_out), display_id=validation_d)

        t1 = time.time()
        epoch_duration.append(t1-t0)

        if log_to_console:
            ema_time = _ema(epoch_duration)
            time_left = (num_epochs - epoch - 1) * ema_time
            epoch_output = f"<b>Epoch [{epoch+1}/{num_epochs}]</b><br>Average Epoch Loss: {np.mean(loss_array[epoch]):.4f}<br>Estimated time left: {time_left:.2f}"
            update_display(HTML(epoch_output), display_id=epoch_d)
        
    if training_params['total_combinations'] != 0:
        training_params['total_combinations'] = training_params['total_combinations'] - 1

    #print("\nTraining complete")
    #print("Final loss: ", np.mean(loss_array[-1]))
    torch.save(model.state_dict(), model_path)
    #print(f"Model parameters saved to {model_path}")
    training_time_end = time.time()
    print("Total time: " + convert_seconds_to_hms((training_time_end - training_time_start) * training_params['total_combinations'] * training_params['iterations']))

    return model, loss_array, val_loss_array