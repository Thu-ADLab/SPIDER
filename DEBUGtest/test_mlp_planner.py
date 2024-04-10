
from spider.data.Dataset import OfflineLogDataset
from spider.planner_zoo.MlpPlanner import MlpPlanner

# import cProfile
# cProfile.run('planner.policy.learn_dataset(100, train_loader=train_loader)')


# setup the planner
planner = MlpPlanner({
    "steps": 20,
    "dt": 0.2,
    "num_object": 5,
    "normalize": False,
    "relative": False,
    "longitudinal_range": (-50, 100),
    "lateral_range": (-20,20),

    "learning_rate": 0.0001,
    "enable_tensorboard": True,
    "tensorboard_root": './tensorboard/'
})

# setup the dataset
dataset = OfflineLogDataset('./dataset/', planner.state_encoder, planner.action_encoder)
train_loader = dataset.get_dataloader(batch_size=64, shuffle=True)  #DataLoader(dataset, batch_size=64, shuffle=True)

# train_mode the planner

planner.policy.learn_dataset(100, train_loader=train_loader)

# save the model
planner.save_state_dict('mlp.pth')

# load the model
planner.load_state_dict('mlp.pth')
# planner.load_state_dict('mlp_good.pth')
# planner.load_state_dict('mlp_best.pth')

# test the planner
test_mode_closed_loop = 0
if test_mode_closed_loop:
    from spider.interface.BaseBenchmark import DummyBenchmark
    benchmark = DummyBenchmark({
    })
    benchmark.test(planner)
else:
    dataset.replay(planner, 0, recording=True)


