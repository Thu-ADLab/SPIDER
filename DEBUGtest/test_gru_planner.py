
from spider.data.Dataset import OfflineLogDataset
from spider.planner_zoo.GRUPlanner import GRUPlanner

# setup the planner
planner = GRUPlanner({
    "steps": 20,
    "dt": 0.2,
    "enable_tensorboard": True,
})

# setup the dataset
dataset = OfflineLogDataset('./dataset/', planner.state_encoder, planner.action_encoder)
train_loader = dataset.get_dataloader(batch_size=64, shuffle=True)  #DataLoader(dataset, batch_size=64, shuffle=True)

# train_mode the planner
planner.policy.learn_dataset(100, train_loader=train_loader)

# save the model
planner.save_state_dict('gru.pth')

# load the model
planner.load_state_dict('gru.pth')

# test the planner
from spider.interface.BaseBenchmark import DummyBenchmark
benchmark = DummyBenchmark({
})
benchmark.test(planner)

