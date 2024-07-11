import utils
from flappy_bird import flappy_bird_gym
import random
import time
import numpy as np

class SmartFlappyBird:
    def __init__(self, iterations, alpha, epsilon, gamma, test_iterations=100):
        self.Qvalues = {}  # Dictionary for Q-values
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.iterations = iterations
        self.test_iterations = test_iterations
        self.train_scores = []
        self.test_scores = []

    def policy(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.get_all_actions())
        else:
            return self.max_arg(state)

    @staticmethod
    def get_all_actions():
        return [0, 1]

    def compute_reward(self, prev_info, new_info, done, observation):
        if done:
            return -10000  # High penalty for dying
        else:
            score_change = new_info['score'] - prev_info['score']
            return score_change * 1000 + 1  # Encourage living longer and scoring

    def get_action(self, state):
        return self.policy(state)

    def state_to_tuple(self, state):
        """Convert numpy array state to a hashable tuple."""
        return tuple(state.flatten())

    def maxQ(self, state):
        actions = self.get_all_actions()
        state_tuple = self.state_to_tuple(state)  # Convert state to tuple
        max_q = float('-inf')
        for action in actions:
            q_value = self.Qvalues.get((state_tuple, action), 0)  # Default to 0 if state-action pair not in Qvalues
            if q_value > max_q:
                max_q = q_value
        return max_q

    def max_arg(self, state):
        actions = self.get_all_actions()
        state_tuple = self.state_to_tuple(state)  # Convert state to tuple
        max_q = float('-inf')
        best_action = None
        for action in actions:
            q_value = self.Qvalues.get((state_tuple, action), 0)  # Default to 0 if state-action pair not in Qvalues
            if q_value > max_q:
                max_q = q_value
                best_action = action
        return best_action

    def update(self, reward, state, action, next_state):
        state_tuple = self.state_to_tuple(state)  # Convert state to tuple
        next_state_tuple = self.state_to_tuple(next_state)  # Convert next state to tuple
        max_q_next = self.maxQ(next_state)
        current_q = self.Qvalues.get((state_tuple, action), 0)  # Default to 0 if state-action pair not in Qvalues
        self.Qvalues[(state_tuple, action)] = current_q + self.alpha * (reward + self.gamma * max_q_next - current_q)

    def run_with_policy(self, train=True):
        env = flappy_bird_gym()  # Initialize your environment
        info = {'score': 0}  # Initialize score info
        observation, _, _ = env.next_frame(0)  # Initial state from the environment
        total_score = 0
        iterations = self.iterations if train else self.test_iterations

        for episode in range(iterations):
            episode_score = 0
            episode_steps = 0
            while True:
                action = self.get_action(observation)
                this_state = observation
                prev_info = info
                observation, reward, done = env.next_frame(action)
                episode_score += reward
                episode_steps += 1
                if train:
                    reward = self.compute_reward(prev_info, info, done, observation)
                    next_state = observation
                    self.update(reward, this_state, action, next_state)
                if done:
                    break
            total_score += episode_score
            print(f"Episode {episode + 1}/{iterations} Score: {episode_score} Steps: {episode_steps}")

        env.close()  # Close environment after training/testing
        return total_score / iterations  # Return average score

    def train(self):
        print("Training...")
        average_train_score = self.run_with_policy(train=True)
        self.train_scores.append(average_train_score)
        print(f"Training Average Score: {average_train_score}")

    def test(self):
        print("Testing...")
        original_epsilon = self.epsilon
        self.epsilon = 0  # Set epsilon to 0 to ensure greedy policy during testing
        average_test_score = self.run_with_policy(train=False)
        self.test_scores.append(average_test_score)
        self.epsilon = original_epsilon  # Restore original epsilon
        print(f"Testing Average Score: {average_test_score}")

    def run(self):
        self.train()
        self.test()

# Main script
if __name__ == "__main__":
    iterations = 300  # Training iterations
    test_iterations = 100  # Testing iterations
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.9
    program = SmartFlappyBird(iterations=iterations, alpha=alpha, epsilon=epsilon, gamma=gamma, test_iterations=test_iterations)
    program.run()

