"""
This script contains the function to train and validate our Transformer model on NACC data.
"""
# import the necessary modules
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import f1_score

import time

from utils import torch_utils

def train(epoch, model, output_size, optimizer, scheduler, criterion, alpha, loader, device, log_interval, log_callback, ckpt_foldername):
    """
    Args:
        - epoch: the current epoch index
        - model: the deep learning model defined in the main.py
        - output_size: the number of classes
        - optimizer: optimizer defined in the main.py
        - scheduler: scheduler defined in the main.py
        - criterion: loss function defined in the main.py
        - alpha: the parameter for L1 regularization
        - loader: data loader defined in the main.py
        - device: cpu or gpu
        - log_interval: the number of batch at which we write the training information into the log file
        - log_callback: the function defined for writing the training information into the log file
        - ckpt_foldername: the checkpoint folder name for storing the .ckpt file for this epoch
    Returns:
        - training_loss: the training loss for this epoch
        - training_macro_f1_score: the training f1_macro_score for this epoch
        - training_micro_f1_score: the training f1_micro_score for this epoch 
        - y_true_array: numpy array storing true labels
        - y_pred_array: numpy array storing predicted labels
        - y_prob_array: numpy array storing probability of positive (think it from the loss function)   
    """
    
    # set up the initialization environment
    start_time = time.time()
    model.train()

    # initialize the statistics
    running_loss = 0.0  # running loss for log_interval
    epoch_loss = 0.0  # loss for this epoch
    y_true_array = np.array([])  # numpy array to store all true labels in this epoch
    y_pred_array = np.array([])  # numpy array to store all predicted labels in this epoch 
    y_prob_array = np.empty((1, output_size))  # numpy array to store all predicted probability of positive label

    # define a sigmoid layer to get compute the correctness
    # sigmoid = nn.Sigmoid()
    # Here should be an softmax layer!!!!!!
    softmax = nn.Softmax(dim=1)

    # the output of the dataloader is (features, labels)
    # the dimension of feature is [num_batch, max_time_steps, feature_dim]
    # labels is a nn.LongTensor type tensor of shape (batch_size,)
    for batch_idx, (features, labels) in enumerate(loader):
        
        features = features.to(device)
        labels = labels.to(device)
        #print(features.shape)
        # input all the input vectors into the model
        preds = model(features)
        
        # compute the loss, here, since the task is to predict whether a patient has
        # Alzheimer or not, we use BCELoss as criterion
        loss = criterion(preds, labels.long())  # redunction = "mean" as default
        # add L1 regularization
        hidden_weights = model.hidden.weight.view(-1)
        output_weights = model.output.weight.view(-1)
        l1_regularization = torch.norm(hidden_weights, 1) + torch.norm(output_weights, 1)
        loss += alpha * l1_regularization

        # compute loss
        running_loss += loss.item()
        epoch_loss += loss.item()
        
        # store true and predictive labels into numpy array
        with torch.no_grad():
            y_true = labels.cpu().numpy()
            y_pred = (torch.argmax(softmax(preds), dim=1)).cpu().numpy()
            y_prob = softmax(preds).cpu().numpy()
            y_true_array = np.concatenate((y_true_array, y_true), axis=0)
            y_pred_array = np.concatenate((y_pred_array, y_pred), axis=0)
            y_prob_array = np.concatenate((y_prob_array, y_prob), axis=0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            # to get the time for log_interval batches
            elapse = time.time() - start_time
            # to write some trainig information into the log file
            log_callback("Epoch: {} \t Training process".format(epoch))
            
            log_callback()
            
            log_callback("Epoch: {0} \t"
                         "Time: {1}s / {2} batches, avg_time: {3}\n".format(
                         epoch, elapse, log_interval, elapse / log_interval ))

            
            log_callback("Train Epoch: {} [{}/{} ({:.0f}%)]\tTraining Loss: {:.6f}".format(
                epoch, batch_idx * len(labels), len(loader.dataset), 100. * batch_idx / len(loader), running_loss / log_interval))
            
            log_callback()

            log_callback("Train Epoch: {} [{}/{} ({:.0f}%)]\tTraining F1 macro: {:.6f}".format(
                epoch, batch_idx * len(labels), len(loader.dataset), 100. * batch_idx / len(loader), f1_score(y_true, y_pred, average='macro')))
        
            log_callback()

            log_callback("Train Epoch: {} [{}/{} ({:.0f}%)]\tTraining F1 micro: {:.6f}".format( epoch, batch_idx * len(labels), len(loader.dataset), 100. * batch_idx / len(loader), f1_score(y_true, y_pred, average='micro')))

            log_callback()

            # reset the start_time
            start_time = time.time()
            # reset the running loss
            running_loss = 0.0

    # to save the training model as a checkpoint 
    # Note here how the scheduler works!!!!!!!!!!!!!
    torch_utils.save(ckpt_foldername + "/linear_regression_NACC_" + str(epoch) + ".cpkt", epoch, model, optimizer, scheduler)    

    # put statistics arrays to a python dictionary
    train_stat = {"train_true": y_true_array, 
                  "train_pred": y_pred_array,
                  "train_prob": y_prob_array[1:, :]}

    # Return the training epoch loss and training epoch F1 macro and training epoch F1 micro score
    return epoch_loss / len(loader), f1_score(y_true_array, y_pred_array, average="macro"), f1_score(y_true_array, y_pred_array, average="micro"), train_stat
    #return epoch_loss / len(loader), f1_score(y_true_array, y_pred_array, average="weighted")

def validation(epoch, model, output_size, criterion, alpha, loader, device, log_callback):
    """
    Args:
        - epoch: the epoch
        - model: deep learning model we want to evaluation
        - output_size: the number of classes
        - criterion: loss function defined in main.py
        - alpha: the parameter for L1 regularization
        - loader: validation data loader 
        - device: cpu or gpu
        - log_callback: the function defined for writing the training information into the log file
    Returns:
        - valid_loss: average validation loss for this epoch
        - valid_f1_macro_score: validation F1 macro score for this epoch
        - valid_f1_micro_score: validation F1 micro score for this epoch
        - valid_stat: python dictionray storing statistics for error analysis
    """
    # intialize the environment
    start_time = time.time()
    model.eval()

    # initialize the statistics
    running_loss = 0.0  # running loss for log_interval
    y_true_array = np.array([])  # numpy array to store all true labels in this epoch
    y_pred_array = np.array([])  # numpy array to store all predicted labels in this epoch 
    y_prob_array = np.empty((1, output_size))  # numpy array to store all predicted probability of positive label
    # to set the sigmoid function to compute the correctness
    #sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)

    # Note here we don't need to keep track of gradients
    with torch.no_grad():
        # the output of the dataloader is (features, labels)
        # the dimension of feature is [num_batch, max_time_steps, feature_dim]
        # labels is a nn.LongTensor type tensor of shape (batch_size,)
        for batch_idx, (features, labels) in enumerate(loader):
            # put features and labels into the device
            features = features.to(device)
            labels = labels.to(device)

            # compute the preds
            preds = model(features)

            # to compute the loss
            loss = criterion(preds, labels.long())  # redunction = "mean" as default 
            # add L1 regularization
            hidden_weights = model.hidden.weight.view(-1)
            output_weights = model.output.weight.view(-1)
            l1_regularization = torch.norm(hidden_weights, 1) + torch.norm(output_weights, 1)
            loss += alpha * l1_regularization
            
            # compute loss
            running_loss += loss.item()

            # store true and predictive labels to numpy array for computing F1 score
            y_true = labels.cpu().numpy()
            y_pred = (torch.argmax(softmax(preds), dim=1)).cpu().numpy()
            y_prob = softmax(preds).cpu().numpy()
            y_true_array = np.concatenate((y_true_array, y_true), axis=0)
            y_pred_array = np.concatenate((y_pred_array, y_pred), axis=0)
            y_prob_array = np.concatenate((y_prob_array, y_prob), axis=0)

        elapse = time.time() - start_time
        
        # write the information into the log file
        log_callback("Epoch: {}\t Validation process".format(epoch))
        
        log_callback()    
        
        log_callback("Epoch: {0} \t"
                     "Time: {1}s / {2} batches, avg_time: {3}\n".format(
                     epoch, elapse, len(loader), elapse / len(loader) ))
        
        log_callback("Train Epoch: {} [{}/{} ({:.0f}%)]\tValidation Loss: {:.6f}".format(
                epoch, batch_idx * len(labels), len(loader.dataset), 100. * batch_idx / len(loader), running_loss / len(loader)))
        
        log_callback()

        log_callback("Train Epoch: {} [{}/{} ({:.0f}%)]\tValidation macro F1: {:.6f}".format(
                epoch, batch_idx * len(labels), len(loader.dataset), 100. * batch_idx / len(loader), f1_score(y_true_array, y_pred_array, average="macro")))
        
        log_callback()

        log_callback("Train Epoch: {} [{}/{} ({:.0f}%)]\tValidation micro F1: {:.6f}".format(
                epoch, batch_idx * len(labels), len(loader.dataset), 100. * batch_idx / len(loader), f1_score(y_true_array, y_pred_array, average="micro")))
        
        log_callback()

        # put statistics arrays to a python dictionary
        valid_stat = {"valid_true": y_true_array, 
                      "valid_pred": y_pred_array,
                      "valid_prob": y_prob_array[1:, :]}

        # return the validation loss and validation F1 score and validation accuracy
        return running_loss / len(loader), f1_score(y_true_array, y_pred_array, average="macro"), f1_score(y_true_array, y_pred_array, average="micro"), valid_stat
        #return running_loss / len(loader), f1_score(y_true_array, y_pred_array, average="weighted")



                     

    
