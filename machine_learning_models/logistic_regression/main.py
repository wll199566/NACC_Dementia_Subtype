# torch
import torch
import torch.nn as nn

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np

# model
#from Transformer.attend_diagnose import make_model
from linear_regression import LinearRegression

# project
from train import train, validation
from evaluation import evaluation
from Dataset import NACCDataset

from utils.config_utils import load_config
from utils.fs_utils import create_folder
from utils.Timer import Timer

# system
import argparse
import time
import pickle

# create folder to store checkpoints
create_folder('checkpoints')
checkPath = 'checkpoints/session_' + Timer.timeFilenameString()
create_folder(checkPath)

# create folder to store models for each epoch
create_folder("models")
modelPath = "models/session_" + Timer.timeFilenameString()
create_folder(modelPath)

# create folder to store log files
create_folder('logs')
logPath = 'logs/log_' + Timer.timeFilenameString()

# create folder to store history dictionary
create_folder("history")
hist_file = "history/session_" + Timer.timeFilenameString() + ".pkl"

# create folder to store statistics dictionary
create_folder("statistics")
stat_path = "statistics/session_" + Timer.timeFilenameString()
stat_train_path = stat_path + "/train"
stat_valid_path = stat_path + "/valid"
create_folder(stat_path)
create_folder(stat_train_path)
create_folder(stat_valid_path)

def append_line_to_log(line = '\n'):
    """
    Append line into the log files.
    """
    with open(logPath, 'a') as f:
        f.write(line + '\n')

def parse_cli(params):
    """
    Parse the arguments for training the model.
    Args:
        - params: parameters read from "config.yaml"
    Returns:
        - args: arguments parsed by parser    
    """
    parser = argparse.ArgumentParser(description="PyTorch Transformer")
    
    # training 
    parser.add_argument('--batch-size', type=int, default=params['batch_size'], metavar='BZ',
                        help='input batch size for training (default: ' + str(params['batch_size']) + ')')
    
    parser.add_argument('--epochs', type=int, default=params['epochs'], metavar='EP',
                        help='number of epochs to train (default: ' + str(params['epochs']) + ')')
    
    # model
    parser.add_argument('--feature_dim', type=int, default=params['feature_dim'], metavar="RD",
                        help="the dimension of features in each time step (default: " + str(params["feature_dim"]) + ")" )
    
    parser.add_argument('--output_dim', type=int, default=params['output_dim'], metavar="DE",
                        help="the dimension of the output layer (default: " + str(params["output_dim"]) + ")" )        

    # hyperparameters
    parser.add_argument('--lr', type=float, default=params['init_learning_rate'], metavar='LR',
                        help='inital learning rate (default: ' + str(params['init_learning_rate']) + ')')

    parser.add_argument('--beta1', type=float, default=params['beta1'], metavar='B1',
                        help=' Adam parameter beta1 (default: ' + str(params['beta1']) + ')')

    parser.add_argument('--beta2', type=float, default=params['beta2'], metavar='B2',
                        help=' Adam parameter beta2 (default: ' + str(params['beta2']) + ')')                    
                        
    parser.add_argument('--epsilon', type=float, default=params['epsilon'], metavar='EL',
                        help=' Adam regularization parameter (default: ' + str(params['epsilon']) + ')')

    parser.add_argument('--seed', type=int, default=params['seed'], metavar='S',
                        help='random seed (default: ' + str(params['seed']) + ')')  

    # system training
    parser.add_argument('--log-interval', type=int, default=params['log_interval'], metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--workers', type=int, default=0, metavar='W',
                        help='workers (default: 0)')

    parser.add_argument('--train_dir', default=params['train_dir'], type=str, metavar='PATHT',
                        help='path to the training files (default: data folder)')

    parser.add_argument('--val_dir', default=params['val_dir'], type=str, metavar='PATHV',
                        help='path to the validation files (default: data folder)')   

    args = parser.parse_args()

    return args     

############################### Main #################################################################

# to get the arguments
params = load_config('config.yaml')
args = parse_cli(params)

# to make everytime the randomization the same
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

np.random.seed(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# to get the directory of training and validation
train_dir = args.train_dir
val_dir = args.val_dir

# to define training and validation dataloader
train_loader = torch.utils.data.DataLoader(NACCDataset(train_dir, "train"), batch_size=args.batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(NACCDataset(val_dir, "valid"), batch_size=args.batch_size, shuffle=False)    
                                                         
# to make the model
#model = make_model(h=args.head_number, d_model=args.d_model, d_ff=args.d_ff, dropout=args.dropout, max_len=args.max_len, record_dim=args.record_dim, d_ff_hidden=args.d_ff_hidden, N=args.encoder_num, de_factor=args.de_factor)
model = LinearRegression(args.feature_dim, args.output_dim) 

# put model into the correspoinding device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# define optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon )

# define scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

# define the criterion
# How to get weights:
# e.g., weight of NORM = (# all training samples) / (# normal samples)
training_loss_weights = [2.488, 43.346, 3.295, 3.683]

weights = torch.FloatTensor(training_loss_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)  # reduction = "mean"

# write the initial information to the log file
append_line_to_log("executing on device: ")
append_line_to_log(str(device))

# to use the inbuilt cudnn auto-tuner to to the best algorithm to use for the hardware
#torch.backends.cudnn.benchmard = True

# to construct dictionary to store the training history
history = {"train_loss":[], "train_f1_macro":[], "train_f1_micro":[], "valid_loss":[], "valid_f1_macro": [], "valid_f1_micro":[]}
#history = {"train_loss":[], "valid_loss":[]}

################################  training process  ############################################################
start_epoch = 1
for epoch in range(start_epoch, args.epochs + 1):
    # train
    train_loss, train_f1_macro, train_f1_micro, train_stat = train(epoch, model, args.output_dim, optimizer, scheduler, criterion, params['alpha'], train_loader, device, args.log_interval, append_line_to_log, checkPath)
    #train_loss, train_f1 = train(epoch, model, optimizer, scheduler, criterion, train_loader, device, args.log_interval, append_line_to_log, checkPath)
    history["train_loss"].append(train_loss)
    history["train_f1_macro"].append(train_f1_macro)
    history["train_f1_micro"].append(train_f1_micro)
    train_stat_filename = stat_train_path+"/train_"+str(epoch)+".pkl"
    with open(train_stat_filename, "wb") as fout:
        pickle.dump(train_stat, fout)

    # validation
    valid_loss, valid_f1_macro ,valid_f1_micro, valid_stat = validation(epoch, model, args.output_dim, criterion, params['alpha'], valid_loader, device, append_line_to_log)
    #valid_loss, valid_f1 = validation(epoch, model, criterion, valid_loader, device, append_line_to_log)
    history["valid_loss"].append(valid_loss)
    history["valid_f1_macro"].append(valid_f1_macro)
    history["valid_f1_micro"].append(valid_f1_micro)
    valid_stat_filename = stat_valid_path+"/valid_"+str(epoch)+".pkl"
    with open(valid_stat_filename, "wb") as fout:
        pickle.dump(valid_stat, fout)

    scheduler.step(train_loss)

    # save the model of this epoch
    model_file = "/model_" + str(epoch) + ".pth"
    model_file = modelPath + model_file
    torch.save(model.state_dict(), model_file)
    append_line_to_log("Save model to " + model_file)

# write the history dictionary to the pickle file
with open(hist_file, "wb") as fout:
    pickle.dump(history, fout)

############################## Find best model ###########################################################
# read in the history file
with open(hist_file, "rb") as fin:
    hist_dict = pickle.load(fin)

# read in validation macro and micro score
f1_macro_valid, f1_micro_valid = hist_dict["valid_f1_macro"], hist_dict["valid_f1_micro"]

# get the epoch for the max valid f1_macro_score and f1_micro_score, respectively
max_valid_f1_macro_epoch = np.argmax(np.array(f1_macro_valid)) +1
max_valid_f1_micro_epoch = np.argmax(np.array(f1_micro_valid)) +1

# print some validation information
print("*"*20 + " Valid Information " + "*"*40 + '\n')
print("max f1_macro epoch: {} | max f1 macro score: {}".format(max_valid_f1_macro_epoch, max(f1_macro_valid)))
print("max f1_micro epoch: {} | max f1 micro score: {}".format(max_valid_f1_micro_epoch, max(f1_micro_valid)))
print('\n')


################################## Evaluation models #############################################3
# initialize a model
eval_model_macro = LinearRegression(params["feature_dim"], params["output_dim"])
eval_model_micro = LinearRegression(params["feature_dim"], params["output_dim"])

# define data loader
eval_dir = params["val_dir"]
eval_loader = torch.utils.data.DataLoader(NACCDataset(eval_dir, "test"), batch_size=16, shuffle=False)

#### macro score ####

# load in the pre-trained model for max f1 macro score
PATH_pretrained_macro = modelPath + "/model_" + str(max_valid_f1_macro_epoch) + ".pth"
print("read in", PATH_pretrained_macro)
eval_model_macro.load_state_dict(torch.load(PATH_pretrained_macro))

# move the model into the corresponding device
eval_model_macro.to(device)

# call the evaluation
print("*"*20 + " Test Max F1 Macro Information " + "*"*40 + '\n')
evaluation(eval_model_macro, params["output_dim"], criterion, params["alpha"], eval_loader, device)
print("\n")


#### micro score ####

# load in the pre-trained model for max f1 micro score
PATH_pretrained_micro = modelPath + "/model_" + str(max_valid_f1_micro_epoch) + ".pth"
print("read in", PATH_pretrained_micro)
eval_model_micro.load_state_dict(torch.load(PATH_pretrained_micro))

# move the model into the corresponding device
eval_model_micro.to(device)

# call the evaluation
print("*"*20 + " Test Max F1 Micro Information " + "*"*40 + '\n')
evaluation(eval_model_micro, params["output_dim"], criterion, params["alpha"], eval_loader, device)
print("\n")
