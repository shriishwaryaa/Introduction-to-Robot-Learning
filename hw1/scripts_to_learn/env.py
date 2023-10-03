import gym

env = gym.make('CartPole-v1', new_step_api=True)
ob = env.reset()

print(ob, type(ob), ob.shape, ob[None], ob[None].shape)