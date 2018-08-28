import sys
sys.path.append('../keras-rl')
import gym
import numpy as np
import time
from rl.util import load_demo_data_from_file, record_demo_data
from copy import deepcopy

def demonstrate(model, env, nb_steps, data_filepath, nb_max_start_steps=4, start_step_policy=None, verbose=1):
    """
    A modified version of the rl.core.test() method that lets an expert agent record demonstrations for a student.
    Outputs a .npy file in the same format as rl.util.record_demo_data()
    """
    if not model.compiled:
        raise RuntimeError('Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
    model.training = False
    model.step = 0

    #matrix of cumulative demo data.
    transitions = []

    #Start a new episode as long as we haven't hit the step limit.
    steps = 0
    while steps < nb_steps:
        model.reset_states()
        observation = deepcopy(env.reset())
        if model.processor is not None:
            observation = model.processor.process_observation(observation)
        assert observation is not None
        # Perform random starts at beginning of episode and do not record them into the demo.
        #This gives the set some variety, even in simple envs like cartpole.
        nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
        for _ in range(nb_random_start_steps):
            if start_step_policy is None:
                action = env.action_space.sample()
            else:
                action = start_step_policy(observation)
            if model.processor is not None:
                action = model.processor.process_action(action)
            observation, r, done, info = env.step(action)
            observation = deepcopy(observation)
            if model.processor is not None:
                observation, r, done, info = model.processor.process_step(observation, r, done, info)
            if done:
                warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                observation = deepcopy(env.reset())
                if model.processor is not None:
                    observation = model.processor.process_observation(observation)
                break
        # Run the episode until we're done.
        done = False
        while not done:
            transition = [observation]
            action = model.forward(observation)
            if model.processor is not None:
                action = model.processor.process_action(action)
            transition.append(action)
            reward = 0.
            accumulated_info = {}
            observation, r, d, info = env.step(action)
            #We add unprocessed reward to demoset, letting us use different reward clipping for the student model.
            transition.append(r)
            transition.append(d)
            if len(transitions) < nb_steps:
                transitions.append(transition)
            observation = deepcopy(observation)
            if model.processor is not None:
                #now we process the reward
                observation, r, d, info = model.processor.process_step(observation, r, d, info)
            if d:
                done = True
            model.backward(reward, terminal=done)
            steps += 1
            model.step += 1

        model.forward(observation)
        model.backward(0., terminal=False)

    model._on_test_end()

    data_matrix = np.array(transitions)
    np.save(data_filepath, data_matrix)

def calc_ep_rs(demo_array):
    """
    Calculate the total episode reward associated with each transition. We use this
    as a naive way to crop the data when using subsets.
    """
    episode_rs = np.zeros(demo_array.shape[0])
    episode_total = 0
    episode_start = 0
    for i, transition in enumerate(demo_array):
        reward = transition[-2]
        reward = np.sign(reward) * np.log(1 + abs(reward))
        episode_total += reward
        if transition[-1]: #terminal
            episode_rs[episode_start : i + 1] = episode_total
            episode_total = 0
            episode_start = i + 1

    return episode_rs

def reward_threshold_subset(demo_array, reward_min):
    rs = calc_ep_rs(demo_array)
    cropped_demos = []
    for i,transition in enumerate(demo_array):
        if rs[i] > reward_min:
            cropped_demos.append(transition)
    return np.array(cropped_demos)

def demo_avg(demo_array):
    demo_array = np.array(list(set(calc_ep_rs(demo_array))))
    return np.mean(demo_array)

if __name__ == "__main__":
    record_demo_data('RocketLander-v0', steps=50000)
    print(len(load_demo_data_from_file()))
