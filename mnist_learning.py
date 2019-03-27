# 自己手写10张图片0到9,打乱顺序,然后利用模型进行识别
# 利用keras的图片预处理,生成一万张图片进行测试
# 学习如何储存训练好的模型
# 用pillow将图片处理成尺寸为(28,28)的灰度图
# 需要遍历文件夹的文件名
import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist
import os
import PIL.Image as image
import matplotlib.pyplot as plt

# 我自己手写的数字的图片绝对路径用列表包含起来
pictures_path = ['/home/ggx/deep_learning/deep_learning_mnist_my_picture/'+i for i 
                    in os.listdir('/home/ggx/deep_learning/deep_learning_mnist_my_picture/')]
my_pictures_train = np.zeros((10,28,28))
print(pictures_path)
my_pictures_labels = np.array([4,3,8,1,0,5,7,6,9,2]).reshape(10,1)
k = 0
for i in pictures_path:
    my_pictures_train[k,:,:] = np.array(image.open(i).resize((28,28)).convert('L'))
    k += 1
my_pictures_train = np.array(my_pictures_train)/255.0
#print(my_pictures_train[0])

#(x_train,y_train),(x_test,y_test) = mnist.load_data()
#x_train,x_test = x_train/255.0,x_test/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(256,activation='relu'), # 不知道这里节点数有没有什么规则
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
                loss = 'sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(my_pictures_train,my_pictures_labels,epochs=10)
#model.evaluate(x_test,y_test)

predictions = model.predict(my_pictures_train)
plt.figure()
k = 1
for i in predictions:
    print(np.argmax(i))
    plt.subplot(3,4,k)
    plt.imshow(my_pictures_train[k-1])
    plt.xlabel('%d'%np.argmax(i))
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    k += 1
plt.show()
