# Dota2Predictor
Predict the outcome of a Dota 2 match based on heroes picked and game duration

## Data Collection
Data sourced from dota2api
DataCollecterFinal.py contains code to pull match data from the dota2api, only games saved are all pick game type with 10 players present (No bots or leavers)
traindata.npy currently contains over 300k matches the script is written so that data can continue to be concatenated onto the end of the array

## Predictor.py
Contains the code for the Neural Network constructed in keras. Also prints out confusion matrix for final trained network and saved the network model. Tensorboard files are also created and stored in Tensorboard folder.
