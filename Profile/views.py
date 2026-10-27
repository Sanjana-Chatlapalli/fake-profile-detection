from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
import os
import pandas as pd
import numpy as np
from django.shortcuts import render
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from django.conf import settings

global model

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def User(request):
    if request.method == 'GET':
       return render(request, 'User.html', {})


def Admin(request):
    if request.method == 'GET':
       return render(request, 'Admin.html', {})

def AdminLogin(request):
    if request.method == 'POST':
      username = request.POST.get('username', False)
      password = request.POST.get('password', False)
      if username == 'admin' and password == 'admin':
       context= {'data':'welcome '+username}
       return render(request, 'AdminScreen.html', context)
      else:
       context= {'data':'login failed'}
       return render(request, 'Admin.html', context)


def importdata():
    dataset_path = os.path.join(settings.BASE_DIR, 'Profile', 'dataset', 'dataset.txt')
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    balance_data = pd.read_csv(dataset_path)
    balance_data = balance_data.abs()
    return balance_data

def splitdataset(balance_data):
    X = balance_data.values[:, 0:8] 
    y_= balance_data.values[:, 8]
    y_ = y_.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(y_)
    print(Y)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
    return train_x, test_x, train_y, test_y

from keras.models import load_model
import numpy as np
import os

def UserCheck(request):
    if request.method == 'POST':
        model_path = os.path.join(settings.BASE_DIR, 'Profile', 'model.h5')

        if not os.path.exists(model_path):
            return render(request, 'User.html', {'data': 'Please Generate Model First'})

        model = load_model(model_path)
        data = request.POST.get('t1')

        try:
            values = list(map(float, data.split(',')))
            if len(values) != 8:
                raise ValueError("Input must have 8 values")

            test = np.array([values])
            prediction = model.predict(test)
            predicted_class = np.argmax(prediction, axis=1)[0]

            msg = "Genuine" if predicted_class == 0 else "Fake"
            msg = "Given Account Details Predicted As " + msg

        except:
            msg = "Invalid input! Enter 8 comma-separated numbers."

        return render(request, 'User.html', {'data': msg})

def GenerateModel(request):
    data = importdata()
    train_x, test_x, train_y, test_y = splitdataset(data)

    model = Sequential()
    model.add(Dense(200, input_shape=(8,), activation='relu', name='fc1'))
    model.add(Dense(200, activation='relu', name='fc2'))
    model.add(Dense(2, activation='softmax', name='output'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)
    
    results = model.evaluate(test_x, test_y)
    ann_acc = results[1] * 100

    # Save model safely
    model_path = os.path.join(settings.BASE_DIR, 'Profile', 'model.h5')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    context = {'data': 'ANN Accuracy : ' + str(ann_acc)}
    return render(request, 'AdminScreen.html', context)

def ViewTrain(request):
    dataset_path = os.path.join(settings.BASE_DIR, 'Profile', 'dataset', 'dataset.txt')

    if not os.path.exists(dataset_path):
        return render(request, 'ViewData.html', {'data': 'Dataset not found!'})

    data = pd.read_csv(dataset_path)
    table = data.to_html(index=False)

    return render(request, 'ViewData.html', {'data': table})
