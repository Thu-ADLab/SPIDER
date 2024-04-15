import spider.visualize as vis
import tqdm

class Trainner:
    '''
    todo:以后加一个把环境打包成gym环境的功能
    '''
    def __init__(self, env_interface, reward_function, visualize=False):
        self.env_interface = env_interface
        self.reward_function = reward_function

        self.max_eps_len = 150

        self.n_epochs = 10
        self._visualize = visualize


    def train(self, planner, train_steps, batch_size=16):
        # todo: 是一个step触发训练，还是一个episode触发训练？
        #  以及一轮训练的次数是1吗？可以参考stable baselines3

        policy = planner.policy
        exp_buffer = planner.exp_buffer

        exp_buffer.apply_to(policy, self.reward_function)  # 开始监听

        obs, done = self.env_interface.reset(), False

        policy.set_exploration(enable=True)

        for i in tqdm.tqdm(range(train_steps)):

            # forward
            plan = planner.plan(*obs) # 监听exp_buffer记录了obs, plan
            self.env_interface.conduct_trajectory(plan)
            obs2 = self.env_interface.wrap_observation()

            # feedback
            reward, done = self.reward_function.evaluate_log(obs, plan, obs2) # 监听exp_buffer记录了reward, done
            policy.try_write_reward(reward, done, i)


            # visualize
            if self._visualize:
                vis.cla()
                vis.lazy_draw(*obs, plan)
                vis.title(f"Step {i}, Reward {reward}")
                vis.pause(0.001)

            if done:
                # 一个episode结束，更新网络参数，学习轨迹
                policy.learn_buffer(exp_buffer, batch_size,self.n_epochs)
                obs = self.env_interface.reset()
                exp_buffer.clear()
            else:
                obs = obs2

        policy._activate_exp_buffer = False
        policy.set_exploration(enable=False)


if __name__ == '__main__':
    from spider.interface import DummyInterface, DummyBenchmark
    from spider.rl.reward.TrajectoryReward import TrajectoryReward
    from spider.planner_zoo.DiscretePPOPlanner import DiscretePPOPlanner

    # presets
    ego_size = (5.,2.)

    # setup env
    env_interface = DummyInterface()

    # setup reward
    reward_function = TrajectoryReward(
        (-10., 280.), (-15, 15), (240., 280.), (-10,10), ego_size
    )

    # setup_planner
    planner = DiscretePPOPlanner({
        "ego_veh_width": ego_size[1],
        "ego_veh_length": ego_size[0],
        "enable_tensorboard": True,
    })

    planner_school = Trainner(env_interface, reward_function, visualize=False)
    planner_school.train(planner, 10000)
    planner.policy.save_model('./ppo.pth')

    planner.policy.load_model('./ppo.pth')
    DummyBenchmark({"save_video": True,}).test(planner)