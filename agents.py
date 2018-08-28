import argparse
import sys
sys.path.append('../keras-rl')
from PIL import Image
import numpy as np
import gym
from keras.models import Model
from keras.layers import Flatten, Input, Dense
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K
from rl.agents.dqn import DQNAgent, DQfDAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import PartitionedMemory, PrioritizedMemory
from rl.core import Processor
from rl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint
from rl.util import load_demo_data_from_file
from record_demonstrations import demonstrate, reward_threshold_subset

parser = argparse.ArgumentParser()
parser.add_argument('--mode',choices=['train','test','demonstrate'],default='test')
parser.add_argument('--model',choices=['student','expert'],default='expert')
args = parser.parse_args()


class RocketProcessor(Processor):
    def process_observation(self, observation):
        return np.array(observation, dtype='float32')

    def process_state_batch(self, batch):
        return np.array(batch).astype('float32')

    def process_reward(self, reward):
        return np.sign(reward) * np.log(1 + abs(reward))

    def process_demo_data(self, demo_data):
        for step in demo_data:
            step[0] = self.process_observation(step[0])
            step[2] = self.process_reward(step[2])
        return demo_data

env = gym.make("LunarLander-v2")
np.random.seed(231)
env.seed(123)
nb_actions = env.action_space.n

WINDOW_LENGTH = 2
input_shape = (WINDOW_LENGTH, 8)

# DQfD "student" model architecture
sensors = Input(shape=(input_shape))
s_dense = Flatten()(sensors)
s_dense = Dense(64, activation='relu', kernel_regularizer=l2(.0001))(s_dense)
s_dense2= Dense(128, activation='relu', kernel_regularizer=l2(.0001))(s_dense)
s_dense3 = Dense(64, activation='relu', kernel_regularizer=l2(.0001))(s_dense2)
s_actions = Dense(nb_actions, activation='linear', kernel_regularizer=l2(.0001))(s_dense3)
student_model = Model(inputs=sensors, outputs=s_actions)
# "Expert" (regular dqn) model architecture
e_dense = Flatten()(sensors)
e_dense = Dense(64, activation='relu')(e_dense)
e_dense2= Dense(128, activation='relu')(e_dense)
e_dense3 = Dense(64, activation='relu')(e_dense2)
e_actions = Dense(nb_actions, activation='linear')(e_dense3)
expert_model = Model(inputs=sensors, outputs=e_actions)

processor = RocketProcessor()
model_saves = './model_saves/'

if __name__ == "__main__":
    if args.model == 'student':
        # load expert data
        expert_demo_data = load_demo_data_from_file(model_saves + 'demos.npy')
        expert_demo_data = reward_threshold_subset(expert_demo_data,0)
        print(expert_demo_data.shape)
        expert_demo_data = processor.process_demo_data(expert_demo_data)
        # memory
        memory = PartitionedMemory(limit=500000, pre_load_data=expert_demo_data, alpha=.6, start_beta=.4, end_beta=.4, window_length=WINDOW_LENGTH)
        # policy
        policy = EpsGreedyQPolicy(.01)
        # agent
        dqfd = DQfDAgent(model=student_model, nb_actions=nb_actions, policy=policy, memory=memory,
                       processor=processor, enable_double_dqn=True, enable_dueling_network=True, gamma=.99, target_model_update=10000,
                       train_interval=1, delta_clip=1., pretraining_steps=15000, n_step=10, large_margin=.8, lam_2=1)

        lr = .00025
        dqfd.compile(Adam(lr), metrics=['mae'])
        weights_filename = model_saves + 'student_lander15k_weights.h5f'
        checkpoint_weights_filename = model_saves +'student_lander15k_weights{step}.h5f'
        log_filename = model_saves + 'student_lander15k_REWARD_DATA.txt'
        callbacks = [TrainEpisodeLogger(log_filename),
                        ModelIntervalCheckpoint(checkpoint_weights_filename, interval=1000000)
                    ]
        if args.mode == 'train':
            dqfd.fit(env, callbacks=callbacks, nb_steps=4250000, verbose=0, nb_max_episode_steps=1500)
            dqfd.save_weights(weights_filename, overwrite=True)
        if args.mode == 'test':
            dqfd.load_weights(model_saves + 'student_lander15k_weights.h5f')
            dqfd.test(env, nb_episodes=12, visualize=True, verbose=2, nb_max_start_steps=30)
        if args.mode == 'demonstrate':
            print("DQfD cannot demonstrate.")

    if args.model == 'expert':
        # memory
        memory = PrioritizedMemory(limit=500000, alpha=.6, start_beta=.4, end_beta=.4, steps_annealed=5000000, window_length=WINDOW_LENGTH)
        # policy
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.02, value_test=.01,
                                      nb_steps=1000000)
        # agent
        dqn = DQNAgent(model=expert_model, nb_actions=nb_actions, policy=policy, memory=memory,
                       processor=processor, enable_double_dqn=True, enable_dueling_network=True, gamma=.99, target_model_update=10000,
                       train_interval=1, delta_clip=1., nb_steps_warmup=50000)

        lr = .00025
        dqn.compile(Adam(lr), metrics=['mae'])
        weights_filename = model_saves + 'expert_lander_weights.h5f'
        checkpoint_weights_filename = model_saves +'expert_lander_weights{step}.h5f'
        log_filename = model_saves + 'expert_lander_REWARD_DATA.txt'
        callbacks = [TrainEpisodeLogger(log_filename),
                        ModelIntervalCheckpoint(checkpoint_weights_filename, interval=1000000)
                    ]
        if args.mode == 'train':
            dqn.fit(env, callbacks=callbacks, nb_steps=4250000, verbose=0, nb_max_episode_steps=1500)
            dqn.save_weights(weights_filename, overwrite=True)
        if args.mode == 'test':
            dqn.load_weights(model_saves + 'expert_lander_weights.h5f')
            dqn.test(env, nb_episodes=5, visualize=True, verbose=2, nb_max_start_steps=30)
        if args.mode == 'demonstrate':
            dqn.load_weights(model_saves + 'expert_lander_weights.h5f')
            demonstrate(dqn, env, 75000, model_saves + 'demos.npy')
