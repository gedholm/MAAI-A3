import random

def make_action(vector):
	a = [0]*4
	intensity = 1
	i = vector.index(1)
	if i<4:
		a[i] = intensity
	elif i==4:
		a[i-1]= 0
	elif 5<=i<9:
		a[i-5]= -intensity
	return(a)

def make_one_hot_from_action(a):
	i = a.index(1)
	return(make_one_hot(i))

def make_one_hot(i):
	v = [0]*9
	v[i] = 1
	return(v)

def make_action_int(i):
	return(make_action(make_one_hot(i)))

def make_random_one_hot():
	i = random.randrange(0, 9)
	return(make_one_hot(i))

def make_random_action():
	return(make_action(make_random_one_hot()))