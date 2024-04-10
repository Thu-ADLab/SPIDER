
from spider.data.Dataset import OfflineLogDataset
from spider.planner_zoo.GRUPlanner import GRUPlanner

train = 0
test_mode_closed_loop = 0

# setup the planner
planner = GRUPlanner({
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
train_loader = dataset.get_dataloader(batch_size=32, shuffle=True)  #DataLoader(dataset, batch_size=64, shuffle=True)

if train:
    # train_mode the planner
    planner.policy.learn_dataset(50, train_loader=train_loader)

    # save the model
    planner.save_state_dict('gru.pth')

# load the model
# planner.load_state_dict('gru.pth')
planner.load_state_dict('gru_best.pth')

# test the planner

if test_mode_closed_loop:
    from spider.interface.BaseBenchmark import DummyBenchmark
    benchmark = DummyBenchmark({
    })
    benchmark.test(planner)
else:
    dataset.replay(planner, 0, recording=True)
