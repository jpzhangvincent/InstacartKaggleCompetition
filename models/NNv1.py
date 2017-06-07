import pandas as pd
import numpy as np
import tflearn as tfl
import random
import sklearn
from sklearn.model_selection import train_test_split
import os
from loss import *

#os.chdir("C:\\Users\\Austin Chi\\Dropbox\\Research Project Austin\\researchgroup\\Austin\\Research")


label = str(random.randint(0,100000000))
data = pd.read_csv('combined.csv',index_col=0)
MODEL_NAME = "Instacart 3 Layer
#PATH = "C:\\Users\\austi\\Desktop\\"


X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size = .2)



test_net = tfl.input_data(shape = [None,16],name = 'input')
test_net = tfl.layers.normalization.batch_normalization(test_net)
                                                                                                               
test_net = tfl.layers.core.fully_connected(test_net,1024,activation='relu',weights_init = 'normal',bias = True,bias_init = 'normal',regularizer = 'L2')
test_net = tfl.layers.core.dropout(test_net,.73)

test_net = tfl.layers.core.fully_connected(test_net,2048,activation='relu',weights_init = 'normal',bias = True,bias_init = 'normal',regularizer = 'L2')
test_net = tfl.layers.core.dropout(test_net,.73)

test_net = tfl.layers.core.fully_connected(test_net,1024,activation='relu',weights_init = 'normal',bias = True,bias_init = 'normal',regularizer = 'L2')
test_net = tfl.layers.core.dropout(test_net,.73)

test_net = tfl.layers.core.fully_connected(test_net,2,activation='sigmoid',bias = True,name = 'To_Output')
#                                           This learning rate seems to marginally help with local minimums
test_net = tfl.layers.estimator.regression(test_net, name = "output",loss = custom_loss_2)

model = tfl.DNN(test_net, tensorboard_verbose = 0,)

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print("Model Loaded")
else:
    model.fit(np.array(X_train) , y_train ,validation_set = (np.array(X_test), y_test),n_epoch = 5000, snapshot_step = 5000, show_metric = True,run_id = MODEL_NAME)
    model.save(PATH + MODEL_NAME)

#Best Accuracy, 74%

