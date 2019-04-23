import math
import random
import numpy as np

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP     = 4
SPEED_KNEE    = 6
LIDAR_RANGE   = 160/SCALE

INITIAL_RANDOM = 5

HULL_POLY =[
    (-30,+9), (+6,+9), (+34,+1),
    (+34,-8), (-30,-8)
    ]
LEG_DOWN = -8/SCALE
LEG_W, LEG_H = 8/SCALE, 34/SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5


def sum_actions(v1, v2, v3, v4):
	action = [0]*4
	action[0] = v1[0] + v2[0] + v3[0] + v4[0]
	action[1] = v1[1] + v2[1] + v3[1] + v4[1]
	action[2] = v1[2] + v2[2] + v3[2] + v4[2]
	action[3] = v1[3] + v2[3] + v3[3] + v4[3]
	return(action)

def make_action_from_one(v, joint):
	action = [0]*4
	intensity = 0.9
	a = v.index(1)
	if a == 0:
		action[joint] = intensity
	elif a == 1:
		action[joint] = 0
	elif a == 2:
		action[joint] = -intensity
	return(action)

def make_one_hots_action(action):
	#print(action)
	ohs = [[0]*3]*4
	ohs = np.array(ohs)
	eps = 0.05
	for i in range(0, len(action)):
		a = action[i]
		if a > eps:
			ohs[i][0] = 1
		elif abs(a) <= eps:
			ohs[i][1] = 1
		elif a < -eps:
			ohs[i][2] = 1
	#print(ohs)
	return ohs[0], ohs[1], ohs[2], ohs[3]


def make_one_hot_int(a):
	one_hot = [0]*3
	one_hot[a] = 1
	return(one_hot)

def run_trials(PG, env):
	final_returns = []
	render = False
	print("Final trials")
	for i in range(0, 100):
		if i == 0:
			reder = True
		else:
			render = False
		final_returns.append(PG.run_simulation(3000, env, render))
	best_final_return = max(final_returns)
	avg_final_return = sum(final_returns)/len(final_returns)
	print("Best final return: ", best_final_return)
	print("Average final return: ", avg_final_return)
	return(final_returns)


def heuristic_walk(s, env, PG, episode):
	env.reset()
	steps = 0
	total_reward = 0
	a = np.array([0.0, 0.0, 0.0, 0.0])
	STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
	SPEED = 0.29  # Will fall forward on higher speed
	state = STAY_ON_ONE_LEG
	moving_leg = 0
	supporting_leg = 1 - moving_leg
	SUPPORT_KNEE_ANGLE = +0.1
	supporting_knee_angle = SUPPORT_KNEE_ANGLE
	max_frames = 500#random.randrange(100, 300)
	while True:
		one_hot_1, one_hot_2, one_hot_3, one_hot_4 = make_one_hots_action(a)
		s, r, done, info = env.step(a)
		s = s[0:14]
		PG.store_transition(s, one_hot_1, one_hot_2, one_hot_3, one_hot_4, r)
		total_reward += r
		# if steps % 20 == 0 or done:
		#     print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
		#     print("step {} total_reward {:+0.2f}".format(steps, total_reward))
		#     print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
		#     print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
		#     print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
		steps += 1

		contact0 = s[8]
		contact1 = s[13]
		moving_s_base = 4 + 5*moving_leg
		supporting_s_base = 4 + 5*supporting_leg

		hip_targ  = [None,None]   # -0.8 .. +1.1
		knee_targ = [None,None]   # -0.6 .. +0.9
		hip_todo  = [0.0, 0.0]
		knee_todo = [0.0, 0.0]

		if state==STAY_ON_ONE_LEG:
		    hip_targ[moving_leg]  = 1.1
		    knee_targ[moving_leg] = -0.6
		    supporting_knee_angle += 0.03
		    if s[2] > SPEED: supporting_knee_angle += 0.03
		    supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
		    knee_targ[supporting_leg] = supporting_knee_angle
		    if s[supporting_s_base+0] < 0.10: # supporting leg is behind
		        state = PUT_OTHER_DOWN
		if state==PUT_OTHER_DOWN:
		    hip_targ[moving_leg]  = +0.1
		    knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
		    knee_targ[supporting_leg] = supporting_knee_angle
		    if s[moving_s_base+4]:
		        state = PUSH_OFF
		        supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
		if state==PUSH_OFF:
		    knee_targ[moving_leg] = supporting_knee_angle
		    knee_targ[supporting_leg] = +1.0
		    if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
		        state = STAY_ON_ONE_LEG
		        moving_leg = 1 - moving_leg
		        supporting_leg = 1 - moving_leg

		if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
		if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
		if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
		if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

		hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
		hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
		knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
		knee_todo[1] -= 15.0*s[3]

		a[0] = hip_todo[0]
		a[1] = knee_todo[0]
		a[2] = hip_todo[1]
		a[3] = knee_todo[1]
		a = np.clip(0.5*a, -1.0, 1.0)
		if episode == 0: 
			env.render()
			#print(sum(PG.episode_rewards))
		if done or steps > max_frames: 
			PG.learn()
			return