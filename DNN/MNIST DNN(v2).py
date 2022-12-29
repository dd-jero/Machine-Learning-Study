# MNIST (손글씨 숫자)를 Deep Neural Network
# tensorflow v2를 이용함

import numpy as np
import random
import tensorflow as tf

# random.seed를 777로 고정한 상태로 알고리즘을 실행하여 난수의 생성 패턴을 매 실행마다 동일하게 관리할 수 있음. 이 코드가 없으면 매 실행마다 다른 결과가 출력
random.seed(777)            
learning_rate = 0.001
training_epochs = 15
batch_size = 100
nb_classes = 10

# tf.keras.datasets.mnist.load_data()를 통해 train data와 test data를 각각 (image, label) 형태로 변환
# 변환된 image와 label은 모두 Numpy array 형식이며 
# 각각의 형태는 image_train.shape (60000,28,28), label_train.shape(60000, ), image_test.shape(10000,28,28), label_test.shape(10000,)
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data() 
print(x_train.shape)

# x_train.shape[0]이 60000 => MNIST data가 총 60000개의 이미지
# reshape()로 각각의 image를 행이 60000, 열이 28*28인 형태로 저장
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)

# tf.keras.utils.to_categorical()로 각각 t_rain, y_test를 nb_calsses(10개)에 맞게 one hot encoding 
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

tf.model = tf.keras.Sequential()    # model을 sequential(순차) 모델로 생성

# model에 add()를 통해 layer층을 추가할 수 있음.
# 아래의 코드로 3개의 hidden layer로 구성되며 input_dim은 입력 뉴런의 수, units은 출력 뉴런의 수, activation은 활성함수를 의미
# ReLU는 backpropagation으로 좋은 성능이 나오기 때문에 대부분 ReLU 사용
tf.model.add(tf.keras.layers.Dense(input_dim=784, units=256, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=256, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=nb_classes, activation='softmax'))         # 마지막 layer에서 softmax를 사용함으로 써 확률 값을 출력하여 분류하기 위함 (또는 sigmoid)
# ==> input layer는 784개의 입력을 받아 256개로 축소하여 ReLU를 통과, 첫 hidden layer는 256개의 입력과 출력으로 ReLU를 한 번더 통과
#     output layer는 10개의 class로 multinomial classification을 한다. (softmax)

# compile()을 통해 train을 위한 model을 구성, cross entropy loss function과 Adam optimizer 이용
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
tf.model.summary()      # 구현한 model을 요약하여 출력

# fit()으로 model training 시작 
tf.model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs)

# predict()로 x_test에 대한 예측 값을 y_predicted에 저장
y_predicted = tf.model.predict(x_test)


for x in range(0,10):
    random_index = random.randint(0, x_test.shape[0]-1)             # random.randint()로 0 ~ 59999에서 선택한 요소를 정수로 반환하여 random_index에 저장
    print("index: ", random_index, 
          "actual y: ", np.argmax(y_test[random_index]),            # y_test의 random_index 행의 가장 큰 원소 반환 (argmax()이용)
          "predicted y: ", np.argmax(y_predicted[random_index]))    

evaluation = tf.model.evaluate(x_test, y_test)          # 학습에서 얻은 model을 test data로 평가한 것을 evaluation으로 반환
print("loss: ", evaluation[0])                          # evaluation[0]은 loss(손실 값)
print("accuracy: ", evaluation[1])                      # evaluation[1]은 accuracy(정확도 값)
