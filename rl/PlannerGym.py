import spider.visualize as vis
import tqdm

'''
qzl: 基本的伪代码

state, action, next_state, done = None, None, None, False
_env.reset()
closed_loop = True # 默认闭环，开环的话不会计算奖励函数并储存经验池，纯做policy.act(state)

while True:
    observation = _env.observe()

    ------------------------- Planner内部 ------------------------------
    next_state = Encoder.encode(observation)

    if closed_loop:
        reward, done = RewardFunction.evaluate(state, action, next_state)
        agent.experience_buffer.record(state, action, reward, next_state, done)
        # 注意，当state是none的时候，reward的计算以及经验池的record都是无效的

    if done:
        state, action, next_state = None, None, None
        plan = None
    else:
        state = next_state
        action = agent.policy.act(state)
        plan = Decoder.decode(action)
    ---------------------------------------------------------------------

    if plan is None: _env.reset()
    else: _env.step(plan)
'''

class PlannerGym:
    def __init__(self, env_interface, reward_function, visualize=False):
        self.env_interface = env_interface
        self.reward_function = reward_function
        self._visualize = visualize


    def train(self, planner, train_steps, batch_size=64):

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
            policy.try_write_reward(reward, i)

            # 学习
            batched_data = exp_buffer.sample(batch_size)
            policy.learn_batch(*batched_data)

            # visualize
            if self._visualize:
                vis.cla()
                vis.lazy_draw(*obs, plan)
                vis.pause(0.001)

            obs = obs2

        policy.set_exploration(enable=False)


if __name__ == '__main__':
    from spider.interface import DummyInterface, DummyBenchmark
    from spider.planner_zoo.DQNPlanner import DQNPlanner
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

    # planner_school = PlannerGym(env_interface, reward_function, visualize=False)
    # planner_school.train(planner_dqn, 5000, 64)
    #
    # planner_dqn.policy.save_model('./q_net.pth')

    planner_dqn.policy.load_model('./q_net.pth')
    DummyBenchmark({"save_video": True,}).test(planner_dqn)
