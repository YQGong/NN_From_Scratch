import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import struct
import copy

# 数据集路径
dataset_path = Path('./dataset')
# 训练图片集路径
train_img_path = './dataset/train-images-idx3-ubyte'
train_lab_path = './dataset/train-labels-idx1-ubyte'
test_img_path = './dataset/t10k-images-idx3-ubyte'
test_lab_path = './dataset/t10k-labels-idx1-ubyte'


def tanh(x):
	return np.tanh(x)

def softmax(x):
	exp = np.exp(x-x.max())
	return exp/exp.sum()
dimensions = [28*28,10]
activation = [tanh,softmax]
distribution=[
{
	'b':[0,0]
},{
	'b':[0,0],
	'w':[-math.sqrt(6/(dimensions[0]+dimensions[1])),math.sqrt(6/(dimensions[0]+dimensions[1]))]
}]

# 初始化参数b
def init_parameters_b(layer):
	dist = distribution[layer]['b']
	return np.random.rand(dimensions[layer])*(dist[1]-dist[0])+dist[0]
# 初始化参数w
def init_parameters_w(layer):
	dist = distribution[layer]['w']
	return np.random.rand(dimensions[layer-1],dimensions[layer])*(dist[1]-dist[0])+dist[0]

#初始化参数方法
def init_parameters():
	parameter=[]
	for i in range(len(distribution)):
		layer_parameter={}
		for j in distribution[i].keys():
			if j=='b':
				layer_parameter['b'] = init_parameters_b(i)
				continue;
			if j=='w':
				layer_parameter['w'] = init_parameters_w(i)
				continue
		parameter.append(layer_parameter)
	return parameter

# 预测函数
def predict(img,init_parameters):
	l0_in = img+parameters[0]['b']
	l0_out = activation[0](l0_in)
	l1_in = np.dot(l0_out,parameters[1]['w'])+parameters[1]['b']
	l1_out = activation[1](l1_in)
	return l1_out
# print(predict(np.random.rand(784),parameters).argmax())
# im = np.reshape(np.random.rand(784),(28,28))
# plt.imshow(im,cmap='gray')
# plt.show()
# print(tanh(0.1))
# print(init_parameters())
# print(softmax(np.array([1,2,3,4])))
# print(init_parameters_w(1).shape)


# 读16个字节
# s = struct.unpack('>4i',train_f.read(16));

# 训练50000个，验证10000个，测试10000个
train_num = 50000
valid_num = 10000
test_num = 10000

# 读入训练图片集和验证图片集
with open(train_img_path,'rb') as f:
	struct.unpack('>4i',f.read(16))
	tmp_img = np.fromfile(f,dtype = np.uint8).reshape(-1,28*28)
	train_img = tmp_img[:train_num]
	valid_img = tmp_img[train_num:]

# 读入测试图片集
with open(test_img_path,'rb') as f:
	struct.unpack('>4i',f.read(16))
	test_img = np.fromfile(f,dtype = np.uint8).reshape(-1,28*28)

# 读入训练标签和验证标签
with open(train_lab_path,'rb') as f:
	struct.unpack('>2i',f.read(8))
	tmp_lab = np.fromfile(f,dtype = np.uint8)
	train_lab = tmp_lab[:train_num]
	valid_lab = tmp_lab[train_num:]

# 读入测试标签
with open(test_lab_path,'rb') as f:
	struct.unpack('>2i',f.read(8))
	test_lab = np.fromfile(f,dtype = np.uint8)

# 展示训练图片
def show_train(index):
	plt.imshow(train_img[index].reshape(28,28),cmap = 'gray')
	print('label  = {}'.format(train_lab[index]))
	plt.show()

# 展示验证图片
def show_valid(index):
	plt.imshow(valid_img[index].reshape(28,28),cmap = 'gray')
	print('label  = {}'.format(valid_lab[index]))
	plt.show()

# 展示测试图片
def show_test(index):
	plt.imshow(test_img[index].reshape(28,28),cmap = 'gray')
	print('label  = {}'.format(test_lab[index]))
	plt.show()



# softmax导数函数
def d_softmax(data):
	sm = softmax(data)
	# diag:对角矩阵  outer：第一个参数挨个乘以第二个参数得到矩阵
	return np.diag(sm)-np.outer(sm,sm)

# tanh导数函数
# def d_tanh(data):
# 	return np.diag(1/(np.cosh(data))**2)
# tanh导数函数优化：
def d_tanh(data):
	return 1/(np.cosh(data))**2

differential = {softmax:d_softmax,tanh:d_tanh}

# lab解析函数
# 将数解析为某一位置为1的一维矩阵
onehot = np.identity(dimensions[-1])

# 求平方差函数
def sqr_loss(img,lab,parameters):
	y_pred = predict(img,parameters)
	y = onehot[lab]
	diff = y-y_pred
	return np.dot(diff,diff)

# 计算梯度
def grad_parameters(img,lab,init_parameters):
	l0_in = img+parameters[0]['b']
	l0_out = activation[0](l0_in)
	l1_in = np.dot(l0_out,parameters[1]['w'])+parameters[1]['b']
	l1_out = activation[1](l1_in)
	
	diff = onehot[lab]-l1_out
	act1 = np.dot(differential[activation[1]](l1_in),diff)

	grad_b1 = -2*act1
	grad_w1 = -2*np.outer(l0_out,act1)
	# 与上文优化d_tanh有关，将矩阵乘法化为数组乘以矩阵
	grad_b0 = -2*differential[activation[0]](l0_in)*np.dot(parameters[1]['w'],act1)

	return {'b1':grad_b1,'w1':grad_w1,'b0':grad_b0}

# test b1
def test_b1(h):
	for i in range(10):
		img_i = np.random.randint(train_num)
		test_parameters = init_parameters()
		derivative = grad_parameters(train_img[img_i],train_lab[img_i],test_parameters)['b1']
		value1 = sqr_loss(train_img[img_i],train_lab[img_i],test_parameters)
		test_parameters[1]['b'][i]+=h
		value2 = sqr_loss(train_img[img_i],train_lab[img_i],test_parameters)
		print(derivative[i]-(value2-value1)/h)

# test b0
def test_b0(h):
	grad_list = []
	for i in range(784):
		img_i = np.random.randint(train_num)
		test_parameters = init_parameters()
		derivative = grad_parameters(train_img[img_i],train_lab[img_i],test_parameters)['b0']
		value1 = sqr_loss(train_img[img_i],train_lab[img_i],test_parameters)
		test_parameters[0]['b'][i]+=h
		value2 = sqr_loss(train_img[img_i],train_lab[img_i],test_parameters)
		grad_list.append(derivative[i]-(value2-value1)/h)
	return grad_list

# test w1
def test_w1(h):
	grad_list = []
	for i in range(784):
		for j in range(10):
			img_i = np.random.randint(train_num)
			test_parameters = init_parameters()
			derivative = grad_parameters(train_img[img_i],train_lab[img_i],test_parameters)['w1']
			value1 = sqr_loss(train_img[img_i],train_lab[img_i],test_parameters)
			test_parameters[1]['w'][i][j]+=h
			value2 = sqr_loss(train_img[img_i],train_lab[img_i],test_parameters)
			grad_list.append(derivative[i][j]-(value2-value1)/h)
	return grad_list

def valid_loss(parameters):
	loss_accu = 0
	for img_i in range(valid_num):
		loss_accu+=sqr_loss(valid_img[img_i],valid_lab[img_i],parameters)
	return loss_accu

# valid 集的精确度
def valid_accuracy(parameters):
	correct = [predict(valid_img[img_i],parameters).argmax()==valid_lab[img_i] for img_i in range(valid_num) ]
	print("validation accuracy:{}".format(correct.count(True)/len(correct)))

# 每组个数
batch_size=100

def train_batch(current_batch,parameters):
	grad_accu = grad_parameters(train_img[current_batch*batch_size+0],train_lab[current_batch*batch_size+0],parameters)
	for img_i in range(1,batch_size):
		grad_tmp = grad_parameters(train_img[current_batch*batch_size+img_i],train_lab[current_batch*batch_size+img_i],parameters)
		for key in grad_accu.keys():
			grad_accu[key] += grad_tmp[key]
	for key in grad_accu.keys():
		grad_accu[key]/=batch_size
	return grad_accu

def combine_parameters(parameters,grad,learn_rate):
	parameter_tmp = copy.deepcopy(parameters)
	parameter_tmp[0]['b'] -= learn_rate*grad['b0']
	parameter_tmp[1]['b'] -= learn_rate*grad['b1']
	parameter_tmp[1]['w'] -= learn_rate*grad['w1']
	return parameter_tmp

def learn_self(learn_rate):
	for i in range(train_num//batch_size):
		if i%100 == 99:
			print("running batch {}/{}".format(i+1,train_num//batch_size))
		grad_tmp = train_batch(i,parameters)
		global parameters
		parameters = combine_parameters(parameters,grad_tmp,learn_rate)

parameters = init_parameters()
valid_accuracy(parameters)
learn_self(1);
valid_accuracy(parameters)

# train_batch(0,parameters)
# print(valid_loss(parameters))
# valid_accuracy(parameters)

# print(np.abs(test_b0(0.000001)).max())
# print(grad_parameters(train_img[0],train_lab[0],parameters))
# print (sqr_loss(train_img[0],train_lab[0],parameters))
# show_train(np.random.randint(train_num))
# show_valid(np.random.randint(valid_num))
# show_test(np.random.randint(test_num))