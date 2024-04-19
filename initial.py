import gymnasium

env = gymnasium.make("ALE/Tetris-v5")
print(env.action_space)
print(env.metadata)
print(env.observation_space)
print(env.reward_range)
print(env.spec)


import math
import torch
from torch.autograd import Variable

class LR_Estimate():
	def __init__(self, n_feat, n_state, n_action, lr=0.05):#n_state - размерность состояний теперь
		self.w, self.b = self.get_gaussian_wb(n_feat,n_state)
		self.n_feat = n_feat
		self.models = []
		self.optimizers = []
		self.criterion = torch.nn.MSELoss()
		for _ in range(n_action):
			model = torch.nn.Linear(n_feat,1)
			self.models.append(model)
			optimizer = torch.optim.SGD(model.parameters(), lr)
			self.optimizers.append(optimizer)

	def get_gaussian_wb(self, n_feat, n_state, sigma=.2):
		torch.manual_seed(0)
		w = torch.randn((n_state,n_feat))/sigma
		b = torch.rand((n_feat))* 2.0 * math.pi
		return w, b

	def get_features(self, s):
		features = (2.0 / self.n_feat) ** 0.5 * torch.cos(torch.matmul(torch.tensor(s).float(), self.w) + self.b)
		return features

	def update(self, s, a, y):
		features = Variable(self.get_features(s))
		y_pred = self.models[a](features)

		loss = self.criterion(y_pred, Variable(torch.Tensor([y])))
		
		self.optimizers[a].zero_grad()
		loss.backward()
		self.optimizers[a].step()

	def predict(self,s):
		features = self.get_features(s)
		with torch.no_grad():
			return torch.tensor([model(features) for model in self.models])

epsilon = 0.03

def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        probs = torch.ones(n_action) * epsilon / n_action
        q_values = estimator.predict(state)
        best_action = torch.argmax(q_values).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function

def q_learning(env, estimator, n_episode, gamma=1.0, epsilon=0.03, epsilon_decay=99):
  for episode in range(n_episode):
    policy = gen_epsilon_greedy_policy(estimator, epsilon * epsilon_decay ** episode, n_action)
    state = env.reset()
    is_done = False
    while not is_done:
        action = policy(state)                                  
        new_state, reward, is_done, info = env.step(action)     
        q_values_next = estimator.predict(new_state)            
        td_target = reward + gamma * torch.max(q_values_next )  
        estimator.update(state, action, td_target)              
        total_reward_episode[episode] += reward                 
        state = new_state

print(env.observation_space.shape[0])
n_state = env.observation_space.shape[0]
n_action = 5
n_feature = 200
lr = 0.03
estimator = LR_Estimate(n_feature, n_state, n_action, lr)

n_episode = 300
total_reward_episode = [0] * n_episode
q_learning(env, estimator, n_episode, epsilon=0.1)

import matplotlib.pyplot as plt
plt.plot(total_reward_episode, 'b.')
plt.title('Зависимость вознаграждения в эпизоде от времени')
plt.xlabel('Эпизод')
plt.ylabel('Полное вознаграждение')
plt.show()
