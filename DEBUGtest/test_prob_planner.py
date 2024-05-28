
from spider.data.Dataset import OfflineLogDataset
from spider.planner_zoo.ProbabilisticPlanner import ProbabilisticPlanner

train = 0
test_mode_closed_loop = 0

# setup the planner
planner = ProbabilisticPlanner({
    "steps": 20,
    "dt": 0.2,
    "num_object": 5,

    "learning_rate": 0.0001,
    "enable_tensorboard": True,
    "tensorboard_root": './tensorboard/'
})

# setup the dataset
dataset = OfflineLogDataset('./dataset_map/', planner.state_encoder, planner.action_encoder, use_cache=True)
train_loader = dataset.get_dataloader(batch_size=32, shuffle=True)  #DataLoader(dataset, batch_size=64, shuffle=True)

if train:
    # train_mode the planner
    planner.policy.learn_dataset(50, train_loader=train_loader)

    # save the model
    planner.save_state_dict('prob.pth')

# load the model
planner.load_state_dict('prob.pth')

# test the planner

if test_mode_closed_loop:
    from spider.interface.BaseBenchmark import DummyBenchmark
    benchmark = DummyBenchmark({
        "map_frequency": 1,
        "save_video": True,
    })
    benchmark.test(planner)
else:
    dataset.replay(planner, 0, recording=True)
