# wheat_nn.py
# Python 3.5.2, TensorFlow 2.1.5, Keras 1.7.0

import numpy as np
import keras as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main():
  # 0. get started
  print("\nWheat dataset using Keras/TensorFlow ")
  np.random.seed(1)

  # 1. load data
  print("Loading wheat data into memory \n")
  train_file = ".\\Data\\wheat_train.txt"
  test_file = ".\\Data\\wheat_test.txt"

  train_x = np.loadtxt(train_file, usecols=range(1,8), delimiter=" ", dtype=np.float32)
  train_y = np.loadtxt(train_file, usecols=[9,10,11], delimiter=" ", dtype=np.float32)
  test_x = np.loadtxt(test_file, usecols=range(1,8), delimiter=" ", dtype=np.float32)
  test_y = np.loadtxt(test_file, usecols=[9,10,11], delimiter=" ", dtype=np.float32)

  # 2. define model
  model = K.models.Sequential()
  model.add(K.layers.Dense(units=20, input_dim=7, activation='tanh'))
  model.add(K.layers.Dense(units=3, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='sgd',  metrics=['accuracy'])

  # 3. train model
  print("Starting training ")
  model.fit(train_x, train_y, batch_size=8, epochs=100, shuffle=True, verbose=0)
  print("Training finished \n")

  # 4. evaluate model
  eval = model.evaluate(test_x, test_y, verbose=0)
  print("Evaluation on test data: loss = %0.6f  accuracy = %0.2f%% \n" % (eval[0], eval[1]*100) )

  # 5. use model
  np.set_printoptions(precision=4, suppress=True)
  unknown = np.array([[20.2400, 16.9100, 0.8897, 6.3150, 3.9620, 5.9010, 6.1880]], dtype=np.float32)
  predicted = model.predict(unknown)
  print("Using model to predict wheat seed variety for features: ")
  print(unknown)
  print("\nPredicted variety is: ")
  print(predicted)

if __name__=="__main__":
  main()
