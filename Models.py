import scipy.io
import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras import optimizers
import keras.backend as K
from sklearn.metrics import confusion_matrix


######################### Loading the Preprocessed Data
#### The MFCCs are preprocessed in MATLAB.
#### Labels are taken as integers.
#### Loading the data
train       = scipy.io.loadmat('dlsp_train.mat')
test        = scipy.io.loadmat('dlsp_test.mat')
train_label = scipy.io.loadmat('dlsp_train_label.mat')
test_label  = scipy.io.loadmat('dlsp_test_label.mat')

X_train = train['data'].T
ytr_all_phoneme = train_label['train_label'][0,0].T
X_test = test['data'].T
yts_all_phoneme = test_label['test_label'][0,0].T
ytr_6 = train_label['train_label'][0,1].T
yts_6 = test_label['test_label'][0,1].T
ytr_3 = train_label['train_label'][0,2].T
yts_3 = test_label['test_label'][0,2].T
ytr_5 = train_label['train_label'][0,3].T
yts_5 = test_label['test_label'][0,3].T
phn_all_phonemes = ('aa', 'ae', 'ah', 'ao', 'aw', 'ax-h', 'ax', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh')
phn_6 = ('VS','NF','SF','WF','ST','CL')
phn_3 = ('SON','OBS','SIL')
phn_5 = ('Vowels','Stops','Fricatives','Nasals','Silences')
nin = X_train.shape[1]


ntr = X_train.shape[0]
nts = X_test.shape[0]
#### Number of train samples : 177080
print('Number of training samples: {0:d}'.format(ntr))

#### Number of test samples : 64145
print('Number of test samples: {0:d}'.format(nts))

#### Number of features in samples : 72
print('Number of features: {0:d}'.format(X_train.shape[1]))

#### Total Number of classes : 61
print('Number of classes: {0:d}'.format((np.unique(yts_all_phoneme)).shape[0]))

############## A class for getting history callback
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        # Create two empty lists, self.loss and self.val_acc
        self.loss = []
        self.val_acc = []
 
    def on_batch_end(self, batch, logs={}):
        # This is called at the end of each batch.  
        # Add the loss in logs.get('loss') to the loss list
        self.loss.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs):
        # This is called at the end of each epoch.  
        # Add the test accuracy in logs.get('val_acc') to the val_acc list
        self.val_acc.append(logs.get('val_acc'))

# Create an instance of the history callback
history_cb = LossHistory()

############## Function to plot validation accuracy
def plotaccuracy(x):
    plt.plot(x)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.show()

############## Function to plot the loss
def plotloss(y):
    plt.semilogy(np.linspace(1,20,44275),y)
    plt.xlabel('Epoch')
    plt.xlim([1,20])
    plt.ylabel('Loss')
    plt.show()

############# Functions to plot Confusion matrices
def Confusion(labels,yhat, phn):
    Matrix = confusion_matrix(labels,yhat)
    Matrixsum = np.sum(Matrix,1)
    Matrix = Matrix/Matrixsum[None,:]
    print(np.array_str(Matrix,precision=3,suppress_small=True))
    fig = plt.figure(figsize = (20,20))
    ax = fig.add_subplot(111)
    im = ax.imshow(Matrix,interpolation='none')
    fig.colorbar(im,ax=ax)
    plt.xlabel('predicted label')
    plt.ylabel('original label')
    xt=plt.xticks(np.arange(np.unique(labels).shape[0]),phn)
    yt=plt.yticks(np.arange(np.unique(labels).shape[0]),phn)


######################### One Layer DNN Classifier for all the Phonemes
#### clear the previous session
K.clear_session()

#### Hidden Layer neurons has 256 units
#### Sigmoid activation function
#### Adam Optimizer with learning rate of 0.001 
nout = int((np.unique(yts_all_phoneme)).shape[0]+1)
DNN_1_all = Sequential()
DNN_1_all.add(Dense(512, input_shape=(nin,), activation='relu', name='hidden'))
DNN_1_all.add(Dense(nout, activation='softmax', name='output'))
opt = optimizers.Adam(lr=0.001)
DNN_1_all.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#### this line gives the summary of the model built above
DNN_1_all.summary()
#### Model is fitted using 25 epochs and also checked for test accuaracy using the validation data
DNN_1_all.fit(X_train, ytr_all_phoneme, epochs=25, batch_size = 100, validation_data=(X_test,yts_all_phoneme), callbacks=[history_cb])

#### Plotting the validation accuracy using the history callback.
plotaccuracy(history_cb.val_acc)

#### Plotting the loss values
plotloss(history_cb.loss)

#### Plotting the confusion matrix
yhat_ts = DNN_1_all.predict(X_test,batch_size=100,verbose=1)
yhat_ts = np.argmax(yhat_ts,axis=1)

Confusion(yts_all_phoneme,yhat_ts,phn_all_phonemes)

######################### One Layer DNN Classifier for Halberstadt 6 groups


#### clear the previous session
K.clear_session()

#### Hidden Layer neurons has 256 units
#### Sigmoid activation function
#### Adam Optimizer with learning rate of 0.001 
nout = int((np.unique(yts_6)).shape[0]+1)
DNN_1_6 = Sequential()
DNN_1_6.add(Dense(512, input_shape=(nin,), activation='relu', name='hidden'))
DNN_1_6.add(Dense(nout, activation='softmax', name='output'))
#### this line gives the summary of the model built above
DNN_1_6.summary()
opt = optimizers.Adam(lr=0.001)
DNN_1_6.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#### Model is fitted using 25 epochs and also checked for test accuaracy using the validation data
DNN_1_6.fit(X_train, ytr_6, epochs=25, batch_size = 100, validation_data=(X_test,yts_6), callbacks=[history_cb])

#### Plotting the validation accuracy using the history callback.
plotaccuracy(history_cb.val_acc)

#### Plotting the loss values
plotloss(history_cb.loss)

#### Plotting the confusion matrix
yhat_ts = DNN_1_6.predict(X_test,batch_size=100,verbose=1)
yhat_ts = np.argmax(yhat_ts,axis=1)

Confusion(yts_6,yhat_ts, phn_6)

######################### One Layer DNN Classifier for Halberstadt 3 groups
K.clear_session()
nout = int((np.unique(yts_3)).shape[0]+1)
DNN_1_3 = Sequential()
DNN_1_3.add(Dense(512, input_shape=(nin,), activation='relu', name='hidden'))
DNN_1_3.add(Dense(nout, activation='softmax', name='output'))
DNN_1_3.summary()
opt = optimizers.Adam(lr=0.001)
DNN_1_3.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
DNN_1_3.fit(X_train, ytr_3, epochs=25, batch_size = 100, validation_data=(X_test,yts_3), callbacks=[history_cb])
plotaccuracy(history_cb.val_acc)
plotloss(history_cb.loss)

yhat_ts = DNN_1_3.predict(X_test,batch_size=100,verbose=1)
yhat_ts = np.argmax(yhat_ts,axis=1)

Confusion(yts_3,yhat_ts, phn_3)

######################### One Layer DNN Classifier for Scalons 5 groups
K.clear_session()
nout = int((np.unique(yts_5)).shape[0]+1)
DNN_1_5 = Sequential()
DNN_1_5.add(Dense(512, input_shape=(nin,), activation='relu', name='hidden'))
DNN_1_5.add(Dense(nout, activation='softmax', name='output'))
DNN_1_5.summary()
opt = optimizers.Adam(lr=0.001)
DNN_1_5.compile(optimizer=opt,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
DNN_1_5.fit(X_train, ytr_5, epochs=25, batch_size = 100, validation_data=(X_test,yts_5), callbacks=[history_cb])
plotaccuracy(history_cb.val_acc)
plotloss(history_cb.loss)

yhat_ts = DNN_1_5.predict(X_test,batch_size=100,verbose=1)
yhat_ts = np.argmax(yhat_ts,axis=1)

Confusion(yts_5,yhat_ts, phn_5)

######################### Two Layer DNN Classifier on all phonemes


K.clear_session()

nout = int((np.unique(yts_all_phoneme)).shape[0]+1)
DNN_2_all = Sequential()
DNN_2_all.add(Dense(512, input_shape=(nin,), activation='relu', name='hidden_1'))
DNN_2_all.add(Dense(512, activation='relu', name='hidden_2'))
DNN_2_all.add(Dense(nout, activation='softmax', name='output'))
DNN_2_all.summary()
opt = optimizers.Adam(lr=0.001)
DNN_2_all.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
DNN_2_all.fit(X_train, ytr_all_phoneme, epochs=25, batch_size = 100, validation_data=(X_test,yts_all_phoneme), callbacks=[history_cb])
plotaccuracy(history_cb.val_acc)
plotloss(history_cb.loss)
yhat_ts = DNN_2_all.predict(X_test,batch_size=100,verbose=1)
yhat_ts = np.argmax(yhat_ts,axis=1)
Confusion(yts_all_phoneme,yhat_ts, phn_all_phonemes)

######################### Two Layer DNN Classifier on Halberstadt 6 groups

K.clear_session()
nout = int((np.unique(yts_6)).shape[0]+1)
DNN_2_6 = Sequential()
DNN_2_6.add(Dense(512, input_shape=(nin,), activation='relu', name='hidden_1'))
DNN_2_6.add(Dense(512, activation='relu', name='hidden_2'))
DNN_2_6.add(Dense(nout, activation='softmax', name='output'))
DNN_2_6.summary()
opt = optimizers.Adam(lr=0.001)
DNN_2_6.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
DNN_2_6.fit(X_train, ytr_6, epochs=25, batch_size = 100, validation_data=(X_test,yts_6), callbacks=[history_cb])
plotaccuracy(history_cb.val_acc)
plotloss(history_cb.loss)
yhat_ts = model.predict(X_test,batch_size=100,verbose=1)
yhat_ts = np.argmax(yhat_ts,axis=1)
Confusion(yts_6,yhat_ts, phn_6)


######################### Two Layer DNN Classifier on Halberstadt 3 groups

K.clear_session()
nout = int((np.unique(yts_3)).shape[0]+1)
DNN_2_3 = Sequential()
DNN_2_3.add(Dense(512, input_shape=(nin,), activation='relu', name='hidden_1'))
DNN_2_3.add(Dense(512, activation='relu', name='hidden_2'))
DNN_2_3.add(Dense(nout, activation='softmax', name='output'))
DNN_2_3.summary()
opt = optimizers.Adam(lr=0.001)
DNN_2_3.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
DNN_2_3.fit(X_train, ytr_3, epochs=25, batch_size = 100, validation_data=(X_test,yts_3), callbacks=[history_cb])
plotaccuracy(history_cb.val_acc)
plotloss(history_cb.loss)
yhat_ts = DNN_2_3.predict(X_test,batch_size=100,verbose=1)
yhat_ts = np.argmax(yhat_ts,axis=1)
Confusion(yts_3,yhat_ts,phn_3)


######################### Two Layer DNN Classifier on Scalons 5 groups

K.clear_session()
nout = int((np.unique(yts_5)).shape[0]+1)
DNN_2_5 = Sequential()
DNN_2_5.add(Dense(512, input_shape=(nin,), activation='relu', name='hidden_1'))
DNN_2_5.add(Dense(512, activation='relu', name='hidden_2'))
DNN_2_5.add(Dense(nout, activation='softmax', name='output'))
DNN_2_5.summary()
opt = optimizers.Adam(lr=0.001)
DNN_2_5.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
DNN_2_5.fit(X_train, ytr_5, epochs=25, batch_size = 100, validation_data=(X_test,yts_5), callbacks=[history_cb])
plotaccuracy(history_cb.val_acc)
plotloss(history_cb.loss)
yhat_ts = DNN_2_5.predict(X_test,batch_size=100,verbose=1)
yhat_ts = np.argmax(yhat_ts,axis=1)
Confusion(yts_5,yhat_ts, phn_5)

######################### Three Layer DNN Classifier on all phonemes

K.clear_session()
nout = int((np.unique(yts_all_phoneme)).shape[0]+1)
DNN_3_all = Sequential()
DNN_3_all.add(Dense(512, input_shape=(nin,), activation='sigmoid', name='hidden_1'))
DNN_3_all.add(Dense(256, activation='sigmoid', name='hidden_2'))
DNN_3_all.add(Dense(512, activation='sigmoid', name='hidden3'))
DNN_3_all.add(Dense(nout, activation='softmax', name='output'))
DNN_3_all.summary()
opt = optimizers.Adam(lr=0.001)
DNN_3_all.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
DNN_3_all.fit(X_train, ytr_all_phoneme, epochs=25, batch_size = 100, validation_data=(X_test,yts_all_phoneme), callbacks=[history_cb])
plotaccuracy(history_cb.val_acc)
plotloss(history_cb.loss)
yhat_ts = DNN_3_all.predict(X_test,batch_size=100,verbose=1)
yhat_ts = np.argmax(yhat_ts,axis=1)
Confusion(yts_all_phoneme,yhat_ts, phn_all_phonemes)

######################### Three Layer DNN Classifier on Halberstadt 6 groups


K.clear_session()
nout = int((np.unique(yts_6)).shape[0]+1)
DNN_3_6 = Sequential()
DNN_3_6.add(Dense(512, input_shape=(nin,), activation='sigmoid', name='hidden_1'))
DNN_3_6.add(Dense(256, activation='sigmoid', name='hidden_2'))
DNN_3_6.add(Dense(512, activation='sigmoid', name='hidden3'))
DNN_3_6.add(Dense(nout, activation='softmax', name='output'))
DNN_3_6.summary()
opt = optimizers.Adam(lr=0.001)
DNN_3_6.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
DNN_3_6.fit(X_train, ytr_6, epochs=25, batch_size = 100, validation_data=(X_test,yts_6), callbacks=[history_cb])
plotaccuracy(history_cb.val_acc)
plotloss(history_cb.loss)
yhat_ts = DNN_3_6.predict(X_test,batch_size=100,verbose=1)
yhat_ts = np.argmax(yhat_ts,axis=1)
Confusion(yts_6,yhat_ts, phn_6)

######################### Three Layer DNN Classifier on Halberstadt 3 groups

K.clear_session()
nout = int((np.unique(yts_3)).shape[0]+1)
DNN_3_3 = Sequential()
DNN_3_3.add(Dense(512, input_shape=(nin,), activation='sigmoid', name='hidden_1'))
DNN_3_3.add(Dense(256,activation='sigmoid', name='hidden_2'))
DNN_3_3.add(Dense(512, activation='sigmoid', name='hidden3'))
DNN_3_3.add(Dense(nout, activation='softmax', name='output'))
DNN_3_3.summary()
opt = optimizers.Adam(lr=0.001)
DNN_3_3.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
DNN_3_3.fit(X_train, ytr_3, epochs=25, batch_size = 100, validation_data=(X_test,yts_3), callbacks=[history_cb])
plotaccuracy(history_cb.val_acc)
plotloss(history_cb.loss)
yhat_ts = DNN_3_3.predict(X_test,batch_size=100,verbose=1)
yhat_ts = np.argmax(yhat_ts,axis=1)
Confusion(yts_3,yhat_ts, phn_3)

######################### Three Layer DNN Classifier on Scalons 5 groups

K.clear_session()
nout = int((np.unique(yts_5)).shape[0]+1)
DNN_3_5 = Sequential()
DNN_3_5.add(Dense(512, input_shape=(nin,), activation='sigmoid', name='hidden_1'))
DNN_3_5.add(Dense(256,activation='sigmoid', name='hidden_2'))
DNN_3_5.add(Dense(512,activation='sigmoid', name='hidden3'))
DNN_3_5.add(Dense(nout, activation='softmax', name='output'))
DNN_3_5.summary()
opt = optimizers.Adam(lr=0.001)
DNN_3_5.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
DNN_3_5.fit(X_train, ytr_5, epochs=25, batch_size = 100, validation_data=(X_test,yts_5), callbacks=[history_cb])
plotaccuracy(history_cb.val_acc)
plotloss(history_cb.loss)
yhat_ts = DNN_3_5.predict(X_test,batch_size=100,verbose=1)
yhat_ts = np.argmax(yhat_ts,axis=1)
Confusion(yts_5,yhat_ts, phn_5)


######################### LSTM + FCN on all phonemes
K.clear_session()

data_dim = 18
timesteps = 4
num_classes = 62
LSTM_all = Sequential()
LSTM_all.add(LSTM(512,input_shape=(timesteps, data_dim))) 
LSTM_all.add(Dropout(0.5))
LSTM_all.add(Dense(512))
LSTM_all.add(Dense(62, activation='softmax'))
LSTM_all.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
X_train = np.reshape(X_train,(X_train.shape[0],4,18))
X_test = np.reshape(X_test,(X_test.shape[0],4,18))
LSTM_all.fit(X_train, ytr_all_phoneme,batch_size=64, epochs=25,validation_data=(X_test, yts_all_phoneme))


######################### on Halberstadt 6 groups


K.clear_session()
LSTM_6 = Sequential()
LSTM_6.add(LSTM(512,input_shape=(timesteps, data_dim)))
LSTM_6.add(Dropout(0.5))
LSTM_6.add(Dense(512))
LSTM_6.add(Dense(7, activation='softmax'))
LSTM_6.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
LSTM_6.fit(X_train, ytr_6,batch_size=64, epochs=25,validation_data=(X_test, yts_6))

######################### on Halberstadt 3 groups

K.clear_session()
LSTM_3 = Sequential()
LSTM_3.add(LSTM(512,input_shape=(timesteps, data_dim)))
LSTM_3.add(Dropout(0.5))
LSTM_3.add(Dense(512)) 
LSTM_3.add(Dense(4, activation='softmax'))
LSTM_3.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
LSTM_3.fit(X_train, ytr_3,batch_size=64, epochs=25,validation_data=(X_test, yts_3))

######################### on Scalons 5 groups

K.clear_session()
LSTM_5 = Sequential()
LSTM_5.add(LSTM(512,input_shape=(timesteps, data_dim)))
LSTM_5.add(Dropout(0.5))
LSTM_5.add(Dense(512))  
LSTM_5.add(Dense(6, activation='softmax'))

LSTM_5.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
LSTM_5.fit(X_train, ytr_5,batch_size=64, epochs=25,validation_data=(X_test, yts_5))

