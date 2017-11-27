import torch 
import torch.nn as nn 
import numpy as np 
import gym 
from torch.autograd import Variable
import torch.optim as optim

def scribe(n,v):
	print(n + ': ' + str(v))

def processState(s): 
	dtype = torch.FloatTensor
	ns = np.array(s).reshape(1,-1)

	res = Variable(torch.from_numpy(ns).type(dtype))
	return res

def discountR(r, y = 0.99): 

	courant = 0
	discounted = np.zeros_like(r)
	for i in reversed(range(r.shape[0])): 
		courant = courant*y + r[i]
		discounted[i] = courant
	return discounted

def culpabilite(valeurs, actions):

	tailleEp = valeurs.shape[0]
	possibilites = valeurs.shape[1] 
	chosen = np.arange(tailleEp)*possibilites
	#scribe('Action', actions)
	#scribe('chosen vefore', chosen)
	for i in range(chosen.shape[0]): 
		chosen[i] += actions[i]
	#scribe('Chosen after', chosen)
	
	return chosen

def guiltyness(model, sH, aH):

	'allo'



env = gym.make('CartPole-v0')
i,h,o = 4, 10, 2


itype = torch.IntTensor
model = nn.Sequential(nn.Linear(i,h), nn.ReLU(), nn.Linear(h,o), nn.Softmax())
lr = 1e-2
adam = optim.Adam(model.parameters(), lr)

updFreq = 3

epochs = 3000
maxMoves = 200

reward_history = []
moves_history = []
	
for epoch in range(epochs):

	ep_hist = []
	s = env.reset()
	complete = False
	reward = 0
	moves = 0

	aH = []
	dH = []
	rH = []
	sH = []

	while not complete: 
		
		sTensor = processState(s)
		distrib = model.forward(sTensor).data.numpy().reshape(-1)
		choice = np.random.choice(distrib, p = distrib)
		action = np.argmax(choice == distrib)

		ns, r, complete, _ = env.step(action)
		reward += r 

		aH.append(action)
		dH.append(distrib)
		rH.append(r)
		sH.append(s)

		
		s = ns 
		moves += 1

		if complete: 

			reward_history.append(reward)
			
			rH = np.array(rH)
			aH = np.array(aH)
			dH = np.array(dH)
			sH = np.vstack(np.array(sH))

			rH = discountR(rH)

			# scribe('actions', aH)
			# scribe('Distrib', dH)
			# scribe('states', sH)

			# input()

			statesTensor = Variable(torch.from_numpy(sH).type(torch.FloatTensor))
			out = model.forward(statesTensor)
			# scribe('Values', out)
			# input()
			#  ----- Calcul des gradients --------

			indexes = culpabilite(out.data, aH)
			# scribe('Respos', indexes)

			flat = out.view(-1).unsqueeze(0)
			# scribe('Not Flat', out)
			# scribe('Flat', flat)
			respos = torch.index_select(flat, 1, Variable(torch.from_numpy(indexes).type(torch.LongTensor)))
			# scribe('respos', respos)
			# input()
			
			rTensor = Variable(torch.from_numpy(rH).type(torch.FloatTensor).unsqueeze(0))
			# scribe('rTensor', rTensor)
			pTensor = -torch.mean(torch.log(respos)*rTensor)
			# scribe('pTensor', pTensor)
			# input()
			pTensor.backward()
			loss = -torch.sum(pTensor)

			#loss.backward()
			
			if epoch % updFreq == 0:
				adam.step()
				adam.zero_grad()

			if epoch % (epochs/10) == 0: 
				meanReward = np.mean(reward_history[-100:])
				text = 'Epochs: ' + str(epoch) + ' | Reward: ' + str(meanReward)
				print (text)
				if meanReward > 175: 
					print ('Solved')
					break

s = env.reset()
while True: 

	sTensor = processState(s)
	distrib = model.forward(sTensor).data.numpy().reshape(-1)
	choice = np.random.choice(distrib, p = distrib)
	action = np.argmax(choice == distrib)

	ns, r, complete,_ = env.step(action)

	s = ns 
	env.render()
	if complete: s = env.reset()




