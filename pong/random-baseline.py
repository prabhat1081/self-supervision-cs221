import gym

def main():
    env = gym.make('Pong-v0')
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
                print("Episode finished after {} timesteps".format(t+1), tot_reward)
                break
            t += 1
        rewards.append(tot_reward)
    print(sum(rewards)/len(rewards))
    env.close()

if __name__ == "__main__":
    main()