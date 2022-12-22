import tensorflow.compat.v1 as tf       # tensorflow 2.0 이상 버전 부터는 session과 placeholder를 사용하지 못한다.
tf.disable_v2_behavior()                # 그래서 compat 모듈을 사용하여 2.0 기능을 끄도록 설정하고 v1을 사용하도록 해야 코드가 동작한다. 

tf.set_random_seed(777)

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data =[[0],[0],[0],[1],[1],[1]]

# placeholder는 tensorflow가 연산을 실행할 때 값을 넣는 공간 (입력될 값의 타입과 size를 미리 정의해두면 후에 연산 과정에서 사용 )
X = tf.placeholder(tf.float32, shape=[None, 2])        
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf. sigmoid(tf.matmul(X,W)+b)
cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))

# tensorflow에서 session은 정의한 노드 및 연산 그래프를 실행할 수 있는 환경을 제공해주는 클래스이다. 
# 연산을 실행하기 위한 리소스를 할당하고 중간 결과 및 변수값을 저장하는 역할을 한다.)
with tf.Session() as sess:                                                  
    sess.run(tf.global_variables_initializer())     # global_variable_initializer로 연산을 시작하기 전 변수들을 초기화 해야

    for step in range(10001):
        cost_val, _ = sess.run([cost,train], feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print(step, cost_val)
    
h, c, a = sess.run([hypothesis, predicted, accuracy], 
                    feed_dict={X:x_data, Y:y_data})

print("\nhypothesis: ", h, "\nCorrect(Y): ",c, "\nAccuracy: ", a)
