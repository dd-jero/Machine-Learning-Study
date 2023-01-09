# tensorflow v2를 사용한 CNN으로 MNIST 분류

import tensorflow as tf
import numpy as np
import random

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# input을 0~255에서 0~1로 정규화(normaliziation)
x_train = x_train/255
x_test = x_test/255

# reshape()를 이용하여 X vector를 [width][height][pixel] 형태로 전환 ( x_train.shape = (50000L,784L)이어서 x_train.shape[0] = 50000)
x_train = x_train.reshape(x_train.shape[0],28,28,1) # MNIST data는 28x28 크기 => width = 28, height = 28 입력, 흑백이므로 1 (컬러라면 3)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

# to_categorical()을 이용하여 one-hot encoding (10개의 class)
y_test = tf.keras.utils.to_categorical(y_test,10)
y_train = tf.keras.utils.to_categorical(y_train, 10)

learning_rate = 0.001
training_epochs = 12
batch_size = 128

# CNN model을 함수 선언
def CNN_model():
    ## model create
    tf.model = tf.keras.Sequential()    # 순차 모델 이용

    # L1: filter 16개, filter size 3x3으로 ReLU 통과 (strides는 default(1)로 설정 => Convolution window 1칸씩 이동)
    tf.model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
    tf.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))     # Max Pooling을 2x2 size로 진행 (2x2내에서 Max값으로 대체)

    # L2: filter 32개, filter size 3x3으로 ReLU 통과 (strides 역시 default)
    tf.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    tf.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))     # L1과 동일한 Max Pooling

    # L3: output 계층으로 classification 위함
    tf.model.add(tf.keras.layers.Flatten())     # Flatten()을 이용해 추출된 주요 특징을  1차원 data로 변환 (이미지 형태의 data를 배열 형태로)
    # activation function으로 softmax classification, Weight Initialization으로 Xaiver Initialization(glorot_normal), output 10개
    tf.model.add(tf.keras.layers.Dense(units=10, kernel_initializer='glorot_normal', activation='softmax')) 

    ## model compile: optimizers는 Adam, loss function으로는 cross entropy 이용
    tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    
    tf.model.summary()   # 생성된 model을 요약하여 터미널에 출력 (layer, output shape, parameter 수 출력)

    return tf.model

## model build
tf.model = CNN_model()

## model fitting(training)
tf.model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

# predict()로 x_test에 대한 예측 값을 y_predicted에 저장
y_predicted = tf.model.predict(x_test)

# random number를 이용해 실제 y와 예측 y를 출력
for x in range(10):
    random_index = random.randint(0, x_test.shape[0]-1)
    print("index: ", random_index,
          "actual y: ", np.argmax(y_test[random_index]),
          "predicted y: ", np.argmax(y_predicted[random_index]))

## evaluation of the model
evaluation = tf.model.evaluate(x_test,y_test)   # evaluation()으로 model의 성능 평가를 evalutation 변수에 저장
print("loss: ", evaluation[0])                  # evaluation[0]은  손실에 대한 정보
print("accuracy: ", evaluation[1])              # evaluation[1]은 정확도에 대한 정보
