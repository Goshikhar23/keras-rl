from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Input, Dense, Activation, LSTM
from keras.optimizers import Adam
import keras.backend as K

# from rl.agents.acer import ACERAgent
# from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
# from rl.memory import SequentialMemory
# from rl.core import Processor
# from rl.callbacks import FileLogger, ModelIntervalCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='CartPole-v1')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--num_agents', type=int, default=1)
args = parser.parse_args()

env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
input_shape = env.observation_space.n

# Make a simple model
# TODO : Check if uint8 works
model_input = Input(shape=(nb_actions,), dtype='uint8', name='main_input')
fc1 = Dense(32, activation = 'relu', name='fc1')(model_input)
lstm = LSTM(32, activation = 'relu', name='lstm')(fc1)
Q = Dense(nb_actions, activation='linear', name='Q')(lstm)
policy = K.clip(Dense(nb_actions, activation='softmax', name='policy')(lstm), min_value=0.00001, max_value=0.999999)
V = K.sum(Q*A, axis=-1)

model = Model(inputs=[model_input], outputs=[policy, Q, V])
avg_model = keras.models.clone_model(model, input_tensors=[model_input])
print('model\n',model.summary())
print('Average Model\n',avg_model.summary())

#Arguements
num_agents = args.num_agents
len_trajectory = 100
max_episode_length = 500
memory_size

acer = ACERAgent(model=model, avg_model=avg_model, num_agents=num_agents,
                 )

acer.compile(lr= , metrics=[])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'acer_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'acer_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'acer_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    acer.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    acer.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    acer.test(env, nb_episodes=10, visualize=False)

elif args.mode == 'test':
    weights_filename = 'acer_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    acer.load_weights(weights_filename)
    acer.test(env, nb_episodes=10, visualize=True)

