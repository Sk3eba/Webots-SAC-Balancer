import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
from controller import Supervisor


#SAC PARAMETERS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99 #farsightness
TAU = 0.005 #target networks
ALPHA = 0.2 #exploration/exploatation, entropy temperature
LR = 3e-4
BATCH_SIZE = 256
BUFFER_SIZE = 100_000


#ACTOR

class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        h = self.net(obs)
        mu = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -20, 2)
        std = log_std.exp()
        dist = Normal(mu, std)
        
        if deterministic: a = mu
        else: a = dist.rsample()
        
        logp = None
        if with_logprob:
            logp = dist.log_prob(a).sum(dim=-1, keepdim=True)
            logp -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=-1, keepdim=True)
        
        return torch.tanh(a), logp


#CRITIC

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


#REPLAY BUFFER
#s -> state
#a -> action
#r -> reward
#ns -> next state
#d -> done

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.stack, zip(*batch))
        return (torch.FloatTensor(s).to(DEVICE), torch.FloatTensor(a).to(DEVICE),
                torch.FloatTensor(r).unsqueeze(1).to(DEVICE), torch.FloatTensor(ns).to(DEVICE),
                torch.FloatTensor(d).unsqueeze(1).to(DEVICE))
    def __len__(self):
        return len(self.buffer)


#environment

class WebotsBalanceEnv:
    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.self_node = self.robot.getSelf()
        self.trans_field = self.self_node.getField("translation")
        self.rot_field = self.self_node.getField("rotation")
        
        self.imu = self.robot.getDevice("imu")
        self.imu.enable(self.timestep)
        self.left_motor = self.robot.getDevice("left_motor")
        self.right_motor = self.robot.getDevice("right_motor")
        
        for m in [self.left_motor, self.right_motor]:
            m.setPosition(float('inf'))
            m.setVelocity(0.0)
            
        self.obs_dim = 3
        self.act_dim = 1

    def reset(self):
        self.trans_field.setSFVec3f([0, 0, 0.15])
        self.rot_field.setSFRotation([0, 0, 1, 0])
        self.self_node.resetPhysics()
        self.robot.step(self.timestep)
        return self._get_obs()

    def _get_obs(self):
        rpy = self.imu.getRollPitchYaw()
        return np.array([rpy[1], 0.0, rpy[0]], dtype=np.float32)

    def step(self, action):
        #wheel power
        torque = float(action[0]) * 3
        self.left_motor.setTorque(torque)
        self.right_motor.setTorque(torque)
        
        if self.robot.step(self.timestep) == -1: return None, 0, True
        
        ns = self._get_obs()
        cp = self.trans_field.getSFVec3f()
        
        is_off_board = abs(cp[1]) > 1.0
        
        v_left = abs(self.left_motor.getVelocity())
        v_right = abs(self.right_motor.getVelocity())
        
        #optional penalties
        velocity_penalty = 0.01 * (v_left + v_right)
        position_penalty = 0.5 * (cp[1]**2)
        
        #reward
        reward = 1.0 - abs(ns[0]) - position_penalty - velocity_penalty
        
        #reset requirements
        done = abs(ns[0]) > 0.4 or cp[2] < 0.1 or is_off_board
        return ns, reward, done


#main loop

env = WebotsBalanceEnv()
actor = SquashedGaussianActor(env.obs_dim, env.act_dim).to(DEVICE)
q1 = QNetwork(env.obs_dim, env.act_dim).to(DEVICE)
q2 = QNetwork(env.obs_dim, env.act_dim).to(DEVICE)
q1_t, q2_t = QNetwork(env.obs_dim, env.act_dim).to(DEVICE), QNetwork(env.obs_dim, env.act_dim).to(DEVICE)
q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())

a_opt = optim.Adam(actor.parameters(), lr=LR)
q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=LR)
buffer = ReplayBuffer(BUFFER_SIZE)

obs = env.reset()
ep_rew, ep_steps, ep_count = 0, 0, 0

while True:
    with torch.no_grad():
        a, _ = actor(torch.FloatTensor(obs).to(DEVICE).unsqueeze(0))
        a = a.cpu().numpy()[0]

    next_obs, reward, done = env.step(a)
    buffer.push(obs, a, reward, next_obs, done)
    obs = next_obs
    ep_rew += reward
    ep_steps += 1

    if len(buffer) > BATCH_SIZE:
        s, act, r, ns, d = buffer.sample(BATCH_SIZE)
        
        with torch.no_grad():
            na, nlogp = actor(ns)
            target_q = r + GAMMA * (1 - d) * (torch.min(q1_t(ns, na), q2_t(ns, na)) - ALPHA * nlogp)

        #critics update
        q_loss = F.mse_loss(q1(s, act), target_q) + F.mse_loss(q2(s, act), target_q)
        q_opt.zero_grad(); q_loss.backward(); q_opt.step()

        #actors update
        na, logp = actor(s)
        a_loss = (ALPHA * logp - torch.min(q1(s, na), q2(s, na))).mean()
        a_opt.zero_grad(); a_loss.backward(); a_opt.step()

        #target soft update
        for p, pt in zip(q1.parameters(), q1_t.parameters()): pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)
        for p, pt in zip(q2.parameters(), q2_t.parameters()): pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)

    if done:
        ep_count += 1
        print(f"Ep: {ep_count} | Steps: {ep_steps} | Reward: {ep_rew:.2f}")
        obs = env.reset(); ep_rew, ep_steps = 0, 0