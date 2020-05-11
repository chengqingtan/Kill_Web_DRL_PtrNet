import torch
import torchvision
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt

def get_2city_distance(n1, n2):
	x1,y1,x2,y2 = n1[0],n1[1],n2[0],n2[1]
	if isinstance(n1, torch.Tensor):
		return torch.sqrt((x2-x1).pow(2)+(y2-y1).pow(2))
	elif isinstance(n1, (list, np.ndarray)):
		return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
	else:
		raise TypeError
	
class Env_tsp():
	def __init__(self, cfg):
		'''
		nodes(cities) : contains all the nodes and their 2 dimensional coordinates 
		[n,2] dimension array e.g. [[0.5,0.7],[0.2,0.3]]
		'''
		self.batch = cfg.batch
		self.city_t = cfg.city_t
		self.xy = cfg.xy
			
	def get_nodes(self, seed = None):
		'''
		return nodes:(city_t,xy)
		'''
		if seed is not None:
			torch.manual_seed(seed)
		return torch.FloatTensor(self.city_t, self.xy).uniform_(0, 1)
		
	def stack_nodes(self):
		'''
		nodes:(city_t,xy)
		return inputs:(batch,city_t,xy)
		'''
		list = [self.get_nodes() for i in range(self.batch)]
		inputs = torch.stack(list, dim = 0)
		return inputs
		
	def stack_random_tours(self):
		'''
		tour:(city_t)
		return tours:(batch,city_t)
		'''
		list = [self.get_random_tour() for i in range(self.batch)]
		tours = torch.stack(list, dim = 0)
		return tours
		
	def stack_l(self, inputs, tours):
		'''
		inputs:(batch,city_t,xy)
		tours:(batch,city_t)
		return l_batch:(batch)
		'''
		list = [self.get_tour_distance(inputs[i], tours[i]) for i in range(self.batch)]
		l_batch = torch.stack(list, dim = 0)
		return l_batch	
		
	def show(self, nodes, tour):
		'''
		plt.annotate
		->arrow goes towards xy=end_pos from xytext=start_pos where comment is put down
		-ref
		http://tanukigraph.hatenablog.com/entry/2017/12/25/224725
		'''
		print('distance:{:.3f}'.format(self.get_tour_distance(nodes, tour)))	
		print(tour)
		
		plt.figure()
		plt.plot(nodes[:,0], nodes[:,1], 'yo', markersize = 16)
		for i in range(self.city_t):
			start_pos = (nodes[tour[i],0], nodes[tour[i],1])
			end_pos	= (nodes[tour[(i+1)%self.city_t],0], nodes[tour[(i+1)%self.city_t],1])
			plt.annotate(str(tour[i].detach().numpy()), xy=end_pos, xycoords='data',
							xytext=start_pos, textcoords='data',
							arrowprops=dict(arrowstyle='<->', connectionstyle='arc3'))
		plt.show()
	
	def shuffle(self, inputs):
		'''
		shuffle nodes order with a set of xy coordinate
		inputs:(batch,city_t,xy)
		return shuffle_inputs:(batch,city_t,xy)
		'''
		shuffle_inputs = torch.zeros(inputs.size())
		for i in range(self.batch):
			perm = torch.randperm(self.city_t)
			shuffle_inputs[i,:,:] = inputs[i,perm,:]
		return shuffle_inputs
		
	def back_tours(self, pred_shuffle_tours, shuffle_inputs, test_inputs):
		'''
		pred_shuffle_tours:(batch,city_t)
		shuffle_inputs:(batch,city_t_t,xy)
		test_inputs:(batch,city_t,xy)
		return pred_tours:(city_t)
		'''
		pred_tours = []
		for i in range(self.batch):
			pred_tour = []
			for j in range(self.city_t):
				xy_temp = shuffle_inputs[i, pred_shuffle_tours[i, j]]
				for k in range(self.city_t):
					if torch.all(torch.eq(xy_temp, test_inputs[i,k])):
						pred_tour.append(torch.tensor(k))
						if len(pred_tour) == self.city_t:
							pred_tours.append(torch.stack(pred_tour, dim = 0)) 
						break
		pred_tours = torch.stack(pred_tours, dim = 0)
		return pred_tours 
			
	def get_tour_distance(self, nodes, tour):
		'''
		nodes:(city_t,xy), tour:(city_t)
		l(= total distance) = l(0-1) + l(1-2) + l(2-3) + ... + l(18-19) + l(19-0) @20%20->0
		return l:(1)
		'''
		l = 0
		for i in range(self.city_t):
			l += get_2city_distance(nodes[tour[i]], nodes[tour[(i+1)%self.city_t]])
		return l

	def get_random_tour(self):
		'''
		return tour:(city_t)
		'''
		tour = []
		while set(tour) != set(range(self.city_t)):
			city = np.random.randint(self.city_t)
			if city not in tour:
				tour.append(city)
		tour = torch.from_numpy(np.array(tour))
		return tour
		
	def get_optimal_tour(self, nodes):
		# dynamic programming algorithm, calculate lengths between all nodes
		points = nodes.numpy()
		all_distances = [[get_2city_distance(x, y) for y in points] for x in points]
		# initial value - just distance from 0 to every other point + keep the track of edges
		A = {(frozenset([0, idx + 1]), idx + 1): (dist, [0, idx + 1]) for idx, dist in enumerate(all_distances[0][1:])}
		cnt = len(points)
		for m in range(2, cnt):
			B = {}
			for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
				for j in S - {0}:
					B[(S, j)] = min([(A[(S - {j}, k)][0] + all_distances[k][j], A[(S - {j}, k)][1] + [j]) for k in S if
									 k != 0 and k != j])  # this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
			A = B
		res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
		tour = torch.from_numpy(np.array(res[1]))
		return tour