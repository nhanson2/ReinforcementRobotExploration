'''Adapted from Ignacio Carlucho'''
import tensorflow.compat.v1 as tf
import numpy as np
import gym
from replay_buffer import ReplayBuffer
from actor import ActorNetwork
from critic import CriticNetwork
from ou_noise import OUNoise
tf.disable_v2_behavior()


ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE =  0.001
TAU = 0.001
ENV_NAME = 'MountainCarContinuous-v0'
RANDOM_SEED = 1234
EXPLORE = 70
DEVICE = '/gpu:0'


def trainer(epochs=1000, MINIBATCH_SIZE=40, GAMMA = 0.99, epsilon=1.0, min_epsilon=0.01, BUFFER_SIZE=10000, train_indicator=True, render=False):
    with tf.Session() as sess:
        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = np.float64(10)
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU, DEVICE)
        critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(), DEVICE)
        rewards = np.zeros(epochs)
        sess.run(tf.global_variables_initializer())
        actor.update_target_network()
        critic.update_target_network()
        replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
        goal = 0
        max_state = -1.
        try:
            critic.recover_critic()
            actor.recover_actor()
            print('********************************')
            print('models restored succesfully')
            print('********************************')
        except :
            print('********************************')
            print('Failed to restore models')
            print('********************************')
        for i in range(epochs):
            state = env.reset()
            state = np.hstack(state)
            ep_reward = 0
            ep_ave_max_q = 0
            done = False
            step = 0
            max_state_episode = -1
            while (not done):
                if render:
                    env.render()
                action = actor.predict(np.reshape(state,(1,state_dim)))
                if np.random.uniform() < epsilon:
                    action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                if train_indicator:
                    replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)), reward,
                                      done, np.reshape(next_state, (actor.s_dim,)))
                    if replay_buffer.size() > MINIBATCH_SIZE:
                        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)
                        target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))
                        y_i = []
                        for k in range(MINIBATCH_SIZE):
                            if t_batch[k]:
                                y_i.append(r_batch[k])
                            else:
                                y_i.append(r_batch[k] + GAMMA * target_q[k])
                        predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))                        
                        ep_ave_max_q += np.amax(predicted_q_value)
                        a_outs = actor.predict(s_batch)
                        grads = critic.action_gradients(s_batch, a_outs)
                        actor.train(s_batch, grads[0])
                        actor.update_target_network()
                        critic.update_target_network()
                state = next_state
                if next_state[0] > max_state_episode:
                    max_state_episode = next_state[0]
                ep_reward = ep_reward + reward
                step +=1
            if done:
                if state[0] > 0.45:
                    goal += 1
            if max_state_episode > max_state:
                max_state = max_state_episode
            print(' training: th',i+1,'// n steps', step,'// R', round(ep_reward,3),'// Pos', round(epsilon,3),'// Eff', round(100.*((goal)/(i+1.)),3), end='\r')
            rewards[i]=ep_reward
        print('*************************')
        print('now we save the model')
        critic.save_critic()
        actor.save_actor()
        print('model saved succesfuly')
        print('*************************')
        return rewards


if __name__ == '__main__':
    rewards = np.zeros((5,200))
    for i in range(5):
        rewards[i] = trainer(epochs=200, epsilon = .01, render = False)
    np.save('rewards_egreedy_5', rewards)
    
