import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import models
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # cifar-10 dataset 가져오기

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("Train samples : ", x_train.shape, y_train.shape)  # 50000개의 32*32, 3개 채널의 train sample
print("Test samples : ", x_test.shape, y_test.shape)  # 10000개의 32*32, 3개 채널의 test sample

print("before normalization")
print("mean : ", np.mean(x_train))
print("std : ", np.std(x_train))

# 정규화 작업
mean = [0, 0, 0]
std = [0, 0, 0]
new_x_train = np.ones(x_train.shape)
new_x_test = np.ones(x_test.shape)

for i in range(3):
    mean[i] = np.mean(x_train[:, :, :, i])
    std[i] = np.std(x_train[:, :, :, i])

for i in range(3):
    new_x_train[:, :, :, i] = x_train[:, :, :, i] - mean[i]
    new_x_train[:, :, :, i] = new_x_train[:, :, :, i] / std[i]
    new_x_test[:, :, :, i] = x_test[:, :, :, i] - mean[i]
    new_x_test[:, :, :, i] = new_x_test[:, :, :, i] / std[i]

x_train = new_x_train
x_test = new_x_test

print("after normalization")
print("mean : ", np.mean(x_train))
print("std : ", np.std(x_train))

# CNN 모델 구현
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

epochs = 10
history = model.fit(x_train, y_train, epochs=epochs)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy : ", test_acc)

predictions = model.predict(x_test)

acc = history.history['accuracy']
loss = history.history['loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()
