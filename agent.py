import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Actor head — outputs raw portfolio weights
        self.actor  = nn.Linear(hidden, action_dim)

        # Critic head — outputs a scalar: "how good is this state?"
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h      = self.shared(x)
        weights = self.actor(h)      
        value   = self.critic(h)     
        return weights, value


# PPO agent 
class PPOAgent:

    def __init__(self, state_dim, action_dim,
                 lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=2):
        """
        lr        : learning rate
        gamma     : discount factor: gamma=0.99 -> reward 100 days later is worth 0.99^100 ≈ 37% now
        clip_eps  : PPO clip threshold (standard = 0.2)
        epochs    : how many gradient steps per batch of experience
        """
        self.gamma    = gamma
        self.clip_eps = clip_eps
        self.epochs   = epochs

        self.net      = PolicyNetwork(state_dim, action_dim)
        self.opt      = optim.Adam(self.net.parameters(), lr=lr)
        self.action_std = 0.02

    #  Act
    def act(self, state, deterministic = False):
        """
        Given a state, return:
          - weights  : raw action (numpy)
          - log_prob : log probability of this action (needed for PPO update)
          - value    : critic's estimate of state value
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)   # add batch dim

        with torch.no_grad():
            weights, value = self.net(state_t)

        # action ~ N(mu, sigma^2)  μ = network output, σ = fixed 0.1
        dist    = torch.distributions.Normal(weights, torch.ones_like(weights) * self.action_std)

        if deterministic:
            action = weights
            log_prob = dist.log_prob(action).sum()
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

        return (action.squeeze().numpy(),
                log_prob.item(),
                value.squeeze().item())

    # Learn
    def learn(self, batch):
        """
        batch : list of (state, action, log_prob, reward, value) tuples

        Steps:
        1. Compute discounted returns G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
        2. Compute advantage           A_t = G_t - V(s_t)
        3. PPO clip loss on actor
        4. MSE loss on critic
        5. Gradient step
        """
        states   = torch.FloatTensor(np.array([b[0] for b in batch]))
        actions  = torch.FloatTensor(np.array([b[1] for b in batch]))
        old_lps  = torch.FloatTensor([b[2] for b in batch])
        rewards  = [b[3] for b in batch]
        values   = torch.FloatTensor([b[4] for b in batch])

        # 1. Discounted returns 
        G, returns = 0, []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # 2. Advantage
        
        # A_t = G_t - V(s_t)
        # Positive/ Negative advantage → action was BETTER than the baseline/ action was WORSE  than the baseline expected
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3+4. PPO Loss 
        for _ in range(self.epochs):
            weights_new, values_new = self.net(states)
            dist    = torch.distributions.Normal(weights_new,
                            torch.ones_like(weights_new) * self.action_std)
            new_lps  = dist.log_prob(actions).sum(dim=1)

            ratio    = torch.exp(new_lps - old_lps) # Probability ratio

            # Clipped surrogate objective
            surr1    = ratio * advantages
            surr2    = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss  = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = nn.MSELoss()(values_new.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)  
            self.opt.step()

        return loss.item()

# Training Loop
def train(env, agent, n_episodes=200, batch_size=128):
    "Each episode - one full pass through the price history (train set)"
    episode_rewards = []

    for ep in range(n_episodes):
        state    = env.reset()
        batch    = []
        ep_reward = 0.0
        done     = False

        decision_state = None
        decision_action = None
        decision_log_prob = None
        decision_value = None
        holding_reward = 0.0
 
        while not done:

            if state is None: 
                break 
            
            if env.should_rebalance_today():
                if decision_state is not None:
                    batch.append((decision_state, decision_action, decision_log_prob, holding_reward, decision_value))
                    holding_reward = 0.0

                    if len(batch) >= batch_size:
                        agent.learn(batch)
                        batch = []

                # new decision
                decision_action, decision_log_prob, decision_value = agent.act(state)
                decision_state = state
                action_to_env = decision_action
            else: 
                action_to_env = None           

            next_state, reward, done = env.step(action_to_env)
            holding_reward += reward
            ep_reward += reward

            if not done:
                state = next_state

        if decision_state is not None:
            batch.append((decision_state, decision_action, decision_log_prob, holding_reward, decision_value))
        if batch:
            agent.learn(batch)

        episode_rewards.append(ep_reward)

        if (ep + 1) % 20 == 0:
            avg = np.mean(episode_rewards[-20:])
            print(f"Episode {ep+1:>4} | Avg Reward (last 20): {avg:+.4f}")

    return episode_rewards
