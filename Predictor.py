from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


my_data = np.load('traindataNew.npy')
#USED TO BE 0:, 1:len(my_data)+1]
d = my_data[0:, 1:len(my_data)+1]
l = my_data[0:, 0]


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
#from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from keras.optimizers import adam



nnscores = []
forestscores = []
svmscores = []
epochs = 50000
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
model.add(Dropout(0.1))
model.add(Dense(500, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(1000, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(300, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
a = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#a = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=a, loss='binary_crossentropy', metrics=['accuracy'])
patience = 100
tbCallBack = TensorBoard(log_dir='C:/Users/baseb/PycharmProjects/Dota2Predictor/', histogram_freq=0, batch_size=1, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
earlyStop = EarlyStopping(monitor='val_acc', patience=patience, min_delta=0, verbose=0, mode='auto')
#class_weight = {0 : 30975., 1: 29290.}

acc = Get_Val_Acc()  # object used to find max acc of training
#removed early stop
callbacks_list = [tbCallBack, earlyStop, acc]
model.fit(xtrain, ytrain_hot, epochs=epochs, callbacks=callbacks_list,  validation_data=(xval, yval_hot),
          batch_size=batchsize, verbose=2, shuffle='true')
score = model.evaluate(xval, yval_hot, verbose=2)


predicted = model.predict(xval)
for i in range(0,len(predicted)):
    if(predicted[i] >= .5):
        predictedlabels.append(1)
    else:
        predictedlabels.append(0)
truelabels = np.append(truelabels, yval)


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
cnf.print_stats()

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
plot_confusion_matrix(cnf_matrix, classes=[0,1,2],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,1,2], normalize=True,
                     title='Normalized confusion matrix')

plt.show()