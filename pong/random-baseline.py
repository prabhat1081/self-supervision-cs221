import gym
import sys

def main(args):
    game = args[0]
    env = gym.make(game)
    print("Game: "+game)
    rewards = []
    for i_episode in range(100):
        observation = env.reset()
        tot_reward = 0
        t = 0
        while(True):
            #env.render()
            #print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            tot_reward += reward
            if done:
                print("Episode {} finished after {} timesteps".format(i_episode, t+1), tot_reward)
                break
            t += 1
        rewards.append(tot_reward)
    print("Mean reward: ", sum(rewards)/len(rewards))
    env.close()

if __name__ == "__main__":
    main(sys.argv[1:])