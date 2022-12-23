# softmax classifier_1

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.set_random_seed(777) # 777을 넣는 이유는 실행 시 마다 같은 초기값을 가지게 하기 위함

x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]] # 4- dimension, 8개의 data samples => shape(8,4)
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]] # 3개의 classes (one hot encoding: class를 0, 1로만 표현) => shape(8,3)

X = tf.placeholder("float", [None,4])
Y = tf.placeholder("float", [None,3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4,nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis = 1)) # axis = 1의 의미는 같은 행에서 열간
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 모든 변수 초기화 

    for step in range(2001):
        _, cost_val = sess.run([optimizer,cost], feed_dict = {X:x_data, Y:y_data})    # data samples을 이용한 연산

        if step % 200 == 0:         # step이 200의 배수일 때는 cost와 step 출력
            print(step, cost_val)
        
    print('----------')
    a = sess.run(hypothesis, feed_dict={X:[[1,11,7,9]]})
    print(a, sess.run(tf.argmax(a,1)))

    print('----------')
    b = sess.run(hypothesis, feed_dict={X:[[1,3,4,3]]})
    print(b, sess.run(tf.argmax(b,1)))

    print('----------')
    c = sess.run(hypothesis, feed_dict={X:[[1,1,0,1]]})
    print(c, sess.run(tf.argmax(c,1)))

    print('----------')
    all = sess.run(hypothesis, feed_dict={X:[[1,11,7,9],[1,3,4,3],[1,1,0,1]]})
    print(all, sess.run(tf.argmax(all,1)))
