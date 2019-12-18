#!/usr/bin/env python
# coding: utf-8

#load模型,并对爬取的教务系统验证码预测，判断准确率，保存预测正确的图片

import pickle
import requests
from matplotlib import pyplot as plt 
from PIL import Image
import numpy as np
from keras.preprocessing import image
import tensorflow as tf 
from keras.models import load_model
import shutil
import random

#包含字符 
captcha_word = "_0123456789abcdefghijklmnopqrstuvwxyz"
#图片的长度和宽度
width = 180
height = 60
#字符数
word_len = 4
#总数
word_class = len(captcha_word)
#验证码素材目录
urp_dataset_dir = 'urp_dataset'

#生成字符索引，同时反向操作一次，方面还原
char_indices = dict((c, i) for i,c in enumerate(captcha_word))
indices_char = dict((i, c) for i,c in enumerate(captcha_word))


#数组转换文字
def vec_to_captcha(vec):
	text = []
	vec = np.reshape(vec,(word_len,word_class))
	
	for i in range(len(vec)):
		temp = vec[i]
		max_index = np.argmax(temp)# 最大值的索引
	
		text.append(captcha_word[max_index % word_class])
		#print(text)
	return ''.join(text)






def my_metrics(y_true, y_pred):
    predict = tf.reshape(y_pred, [-1, word_len, word_class])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(y_true, [-1, word_len,word_class]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    _result = tf.map_fn(fn=lambda e: tf.reduce_all(e),elems=correct_pred,dtype=tf.bool)
    return tf.reduce_mean(tf.cast(_result, tf.float32))






url1='http://219.148.85.172:9111/img/captcha.jpg?15'
url2 = 'http://219.148.85.172:9111/j_spring_security_check'


def urp_predict_getdata(number=100):
	print("urp predict getdata begin")
	
	s=requests.session()

	temp = np.zeros( (height, width, 3),dtype = np.uint8)

	#总共预测数量
	all_code = 100

	#预测结果比例
	all_results=[]

	for j in range(number):

		if j % 20 ==0:
			model = load_model("output/urp_captcha_model.h5",custom_objects={'my_metrics': my_metrics})
		
		#预测正确数量
		right_code =0
		for i in range(0,all_code):
			r1 = s.get(url1)
			
			img_path = 'urp_test.jpg'
			img_path = str(i)+'.jpg'
			with open(img_path, 'wb') as f:
				for chunk in r1.iter_content(chunk_size=1024):
					if chunk:
						f.write(chunk)
						f.flush()
				f.close()

			X = np.zeros((1, height, width, 3), dtype = np.uint8)

			#读取图片
			raw_img = image.load_img(img_path, target_size=(height, width))
			#讲图片转为np数组
			X[0] = image.img_to_array(raw_img)
			
			result = model.predict(X)
			vex_test = vec_to_captcha(result[0])
			#print(img_path,vex_test)
			
			data = {'j_username':random.randint(1,100000),'j_password':random.randint(1,100000),'j_captcha':vex_test}
			
			r2 = s.post(url2,data=data,allow_redirects=False)
			code = str(r2.headers)
			#print(code)
			#print(r2.text)
			#在有的教务系统里面，如果badCredentials在Location里面，说明验证码正确，用户名密码错误
			if 'badCredentials'  in code:
				right_code=right_code+1
				#复制验证码
				targetFile = urp_dataset_dir+"/"+vex_test+"_"+str(random.randint(1,1000000000))+".jpg"
				file = open(targetFile, "wb")
				file.write(open(img_path, "rb").read())
				file.close()
		results = "准确率："+str(right_code/all_code)+"\n"
		print(results)
		all_results.append(results)

	#for i in range(len(all_results)):
	#	print(all_results[i])
	
	print("urp predict getdata end")


if __name__ == "__main__":
	urp_predict_getdata()

