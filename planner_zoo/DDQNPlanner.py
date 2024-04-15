

from spider.rl.policy.DDQNPolicy import DDQNPolicy

from spider.planner_zoo.DQNPlanner import DQNPlanner, MLP_q_network

class DDQNPlanner(DQNPlanner):
    def __init__(self, config=None):
        super().__init__(config)

        self.policy = DDQNPolicy(
            MLP_q_network(self.state_encoder.state_dim, self.action_decoder.action_dim),
            self.action_decoder.action_dim,
            lr=self.config["learning_rate"],
            enable_tensorboard=self.config["enable_tensorboard"],
            tensorboard_root=self.config["tensorboard_root"]
        )


