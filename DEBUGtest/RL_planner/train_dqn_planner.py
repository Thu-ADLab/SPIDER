import spider.visualize as vis
import tqdm

class Trainner:
    '''
    todo:以后加一个把环境打包成gym环境的功能
    '''
    def __init__(self, env_interface, reward_function, visualize=False):
        self.env_interface = env_interface
        self.reward_function = reward_function
        self._visualize = visualize


    def train(self, planner, train_steps, batch_size=64):
        # todo: 是一个step触发训练，还是一个episode触发训练？
        #  以及一轮训练的次数是1吗？可以参考stable baselines3

        policy = planner.policy
        exp_buffer = planner.exp_buffer

        exp_buffer.apply_to(policy, self.reward_function)  # 开始监听

        obs, done = None, True

        policy.set_exploration(enable=True)

        for i in tqdm.tqdm(range(train_steps)):
            if done:
                obs = self.env_interface.reset()

            # forward
            plan = planner.plan(*obs) # 监听exp_buffer记录了obs, plan
            self.env_interface.conduct_trajectory(plan)
            obs2 = self.env_interface.wrap_observation()

            # feedback
            reward, done = self.reward_function.evaluate_log(obs, plan, obs2) # 监听exp_buffer记录了reward, done
            policy.try_write_reward(reward, done, i)

            # 学习
            batched_data = exp_buffer.sample(batch_size)
            policy.learn_batch(*batched_data)

            # visualize
            if self._visualize:
                vis.cla()
                vis.lazy_draw(*obs, plan)
                vis.title(f"Step {i}, Reward {reward}")
                vis.pause(0.001)

            obs = obs2

        policy.set_exploration(enable=False)
if __name__ == '__main__':
    from spider.interface import DummyInterface, DummyBenchmark
    from spider.planner_zoo.DQNPlanner import DQNPlanner
    from spider.planner_zoo.DDQNPlanner import DDQNPlanner
    from spider.rl.reward.TrajectoryReward import TrajectoryReward

    # presets
    ego_size = (5.,2.)

    # setup env
    env_interface = DummyInterface()

    # setup reward
    reward_function = TrajectoryReward(
        (-10., 280.), (-15, 15), (240., 280.), (-10,10), ego_size
    )

    # setup_planner
    planner_dqn = DQNPlanner({
        "ego_veh_width": ego_size[1],
        "ego_veh_length": ego_size[0],
        "enable_tensorboard": True,
    })

    planner_school = Trainner(env_interface, reward_function, visualize=False)
    planner_school.train(planner_dqn, 50000, 64)
    planner_dqn.policy.save_model('./q_net.pth')

    planner_dqn.policy.load_model('./q_net.pth')
    DummyBenchmark({"save_video": True,}).test(planner_dqn)