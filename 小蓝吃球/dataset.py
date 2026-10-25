"""豆子回归、二分类及螺旋分类练习的人工数据生成函数。"""

import random

import numpy as np

def get_beans(counts):
	xs = np.random.rand(counts)
	xs = np.sort(xs)
	ys = [1.2*x+np.random.rand()/10 for x in xs]
	return xs,ys

def get_beans2(counts):
	xs = np.random.rand(counts)
	xs = np.sort(xs)
	ys = np.array([(0.7*x+(0.5-np.random.rand())/5+0.5) for x in xs])
	return xs,ys

def get_beans3(counts):
	xs = np.random.rand(counts)
	xs = np.sort(xs)
	ys = np.zeros(counts)
	for i in range(counts):
		x = xs[i]
		yi = 0.7*x+(0.5-np.random.rand())/50+0.5
		if yi > 0.8:
			ys[i] = 1
	return xs,ys

def get_beans4(counts):
	xs = np.random.rand(counts)*2
	xs = np.sort(xs)
	ys = np.zeros(counts)
	for i in range(counts):
		x = xs[i]
		yi = 0.7*x+(0.5-np.random.rand())/50+0.5
		if yi > 0.8 and yi < 1.4:
			ys[i] = 1


	return xs,ys

def get_beans5(counts):
	xs = np.random.rand(counts,2)*2
	ys = np.zeros(counts)
	for i in range(counts):
		x = xs[i]
		if (x[0]-0.5*x[1]-0.1)>0:
			ys[i] = 1
	return xs,ys

def get_beans6(counts):
	"""合并两类相差半圈的螺旋，返回坐标 X 和 0/1 标签 Y。

	每类生成 int(counts / 2) 个样本，counts 为奇数时实际总数为 counts - 1。
	"""
	posX,posY = genSpiral(int(counts/2),0,1)
	negX,negY = genSpiral(int(counts/2),np.pi,0)
	X = np.vstack((posX,negX))
	Y = np.hstack((posY,negY))
	return X,Y

def genSpiral(counts,deltaT, label):
	"""生成一条螺旋，deltaT 为角度偏移，各坐标添加 [-0.1, 0.1] 的随机噪声。"""
	X = np.zeros((counts,2))
	Y = np.zeros(counts)
	for i in range(counts):
		r = i / counts * 5
		t = 1.75 * i / counts * 2 * np.pi + deltaT;
		x1 = r * np.sin(t) + random.uniform(-0.1,0.1)
		x2 = r * np.cos(t) + random.uniform(-0.1,0.1)
		X[i] = np.array([x1,x2])
		Y[i] = label
	return X,Y
