import numpy as np
import tensorflow as tf
import os

config = tf.ConfigProto(allow_soft_placement=False)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# traindataNew.npy has all 10 heros
my_data = np.load('traindata.npy')
#USED TO BE 0:, 1:len(my_data)+1]
#Pull all data including fb time and total time
#d = my_data[0:, 1:len(my_data)+1]

#Pull just heros
#np.random.shuffle(my_data)
d = my_data[0:, 1:-1]
d = d[:, :-1]
l = my_data[0:, 0]

from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
#from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from keras.optimizers import adam
from sklearn.utils import class_weight


nnscores = []
epochs = 2000
batchsize = 20000

truelabels = []
predictedlabels = []

########################################################################################################################

# Generate batches from indices

#length of data *.7
train_indices = int(len(d)*.7)
xtrain, xval = d[0:train_indices], d[train_indices+1:len(d)+1]
ytrain, yval = l[0:train_indices], l[train_indices+1:len(d)+1]
ytrain_hot = np_utils.to_categorical(ytrain)
yval_hot = np_utils.to_categorical(yval)
ytrain_hot = ytrain
yval_hot = yval


#######################################################################################################################
class Get_Val_Acc(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('val_acc'))
        self.max_acc = max(self.losses)


model = Sequential()
model.add(Dense(1000, input_shape=(len(xtrain[0, :]),), activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(500, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(1000, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(300, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
a = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#a = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=a, loss='binary_crossentropy', metrics=['accuracy'])
patience = 100

directory = 'E:/Documents\GitHub/Dota2Predictor/TensorBoard/Heroes/5 Layer Architecture'
if not os.path.exists(directory):
    os.makedirs(directory)

tbCallBack = TensorBoard(log_dir=directory, histogram_freq=0, write_graph=True, write_grads=False, write_images=False)
earlyStop = EarlyStopping(monitor='val_acc', patience=patience, min_delta=0, verbose=2, mode='auto')

class_weight = class_weight.compute_class_weight('balanced', np.unique(ytrain), ytrain)
class_weight_dict = dict(enumerate(class_weight))

acc = Get_Val_Acc()  # object used to find max acc of training
#callbacks_list = [tbCallBack, earlyStop, acc]
callbacks_list = [tbCallBack, acc]
model.fit(xtrain, ytrain_hot, epochs=epochs, callbacks=callbacks_list, class_weight=class_weight, validation_data=(xval, yval_hot),
          batch_size=batchsize, verbose=2, shuffle='true')
score = model.evaluate(xval, yval_hot, verbose=2)


predicted = model.predict(xval)
for i in range(0,len(predicted)):
    if(predicted[i] >= .5):
        predictedlabels.append(1)
    else:
        predictedlabels.append(0)
truelabels = np.append(truelabels, yval)

model.save('Heroes.h5')

# evaluate the model
#print("max validation %s: %.2f%%" % (model.metrics_names[1], acc.max_acc * 100))
nnscores.append(acc.max_acc * 100)
print('Neural Network:')
print("%.2f%% (+/- %.2f%%)" % (np.mean(nnscores), np.std(nnscores)))



predictedlabels = np.asarray(predictedlabels)
truelabels = truelabels

print('predicted labels: ', predictedlabels)
print('true labels: ', truelabels)

####################################################################################################################

from pandas_ml import ConfusionMatrix

cnf = ConfusionMatrix(truelabels, predictedlabels)
import matplotlib.pyplot as plt


#####################################################################################################################
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(truelabels, predictedlabels)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,1],
                      title='Confusion Matrix - Heroes')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,1], normalize=True,
                     title='Normalized Confusion Matrix - Heroes')

plt.show()