import numpy as np
import glob as gb
import cv2

"""
    ---- Preparing data ---- 
"""
# Loading data
images_path = gb.glob('Hoda 0-9/' + "*.bmp" )

# Feature Extraction
X = []
Y = []
for i in range(len(images_path)):
    path = images_path[i]
    idx = path.rfind('_') - 1
    Y.append(int(path[idx]))
    img = cv2.imread(path, 0)
    img = cv2.resize(img , dsize=(10,10))
    ftrs = np.reshape(img , newshape=(100,))
    X.append(ftrs)

X = np.array(X)
Y = np.array(Y)

    cv2.imshow('image' , img)
     ch = cv2.waitKey(0)
     if ch == ord('q'):
            break
cv2.destroyAllWindows()



# Shuffle data
list_per = np.random.permutation(len(X))
X_per = []
Y_per = []
for i in range(len(list_per)):
    idx_per = list_per[i]
    ftr = X[idx_per]
    lbl = Y[idx_per]
    X_per.append(ftr)
    Y_per.append(lbl)
X_per = np.array(X_per)
Y_per = np.array(Y_per)

# Splitting data to train and validation
trn_tst_split = int(0.7 * len(X_per))
X_train = X_per[0:trn_tst_split, :]
Y_train = Y_per[0:trn_tst_split]

X_val = X_per[trn_tst_split:, :]
Y_val = Y_per[trn_tst_split: ]

# Normalize data
X_train = X_train/255
X_val = X_val / 255

"""
    ---- Design of Neural network ----
"""
from sklearn.neural_network import MLPClassifier

# create neural network
mlp = MLPClassifier(hidden_layer_sizes=(50, 20), activation='logistic',
                    solver='adam', batch_size=50, learning_rate='adaptive',
                    learning_rate_init=0.001, max_iter=200, shuffle=True,
                    tol=0.0000001, verbose=True, momentum=0.95)
# train neural network
mlp.fit(X_train, Y_train)

print('train accuracy : ', mlp.score(X_train, Y_train))
print('val accuracy : ', mlp.score(X_val, Y_val))

loss = mlp.loss_curve_

import matplotlib.pyplot as plt
plt.plot(loss, label= 'loss-Train')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

from sklearn.metrics import confusion_matrix as cm 
Y_val_pred = mlp.predict(X_val)
mlp_cm = cm(Y_val , Y_val_pred)

from sklearn.metrics import classification_report 
mlp_report = classificatiion_report(Y_val , Y_val_pred)

import joblib 
joblib.dump(mlp , 'mlp-network.joblib')
new_model = joblib.load('mlp-network.joblib')
Y_val_pred_newModel = new_model.predict(X_val)





