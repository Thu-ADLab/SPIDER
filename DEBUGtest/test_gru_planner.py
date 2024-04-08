
from spider.data.Dataset import OfflineLogDataset
from spider.planner_zoo.GRUPlanner import GRUPlanner

train = 0
test_mode_closed_loop = 0

# setup the planner
planner = GRUPlanner({
    "steps": 20,
    "dt": 0.2,
    "enable_tensorboard": True,
})

# setup the dataset
dataset = OfflineLogDataset('./dataset/', planner.state_encoder, planner.action_encoder)
train_loader = dataset.get_dataloader(batch_size=64, shuffle=True)  #DataLoader(dataset, batch_size=64, shuffle=True)

if train:
    # train_mode the planner
    planner.policy.learn_dataset(100, train_loader=train_loader)

    # save the model
    planner.save_state_dict('gru.pth')

# load the model
planner.load_state_dict('gru.pth')

# test the planner

if test_mode_closed_loop:
    from spider.interface.BaseBenchmark import DummyBenchmark
    benchmark = DummyBenchmark({
    })
    benchmark.test(planner)
else:
    import spider.visualize as vis
    vis.figure(figsize=(12, 8))
    for i in range(100):
        t, obs, traj = dataset.get_record(i)[:3]
        ##################
        # planner.policy.learn_dataset(1, train_loader=train_loader)
        # b_s, b_a = dataset[i][:2]
        # b_s, b_a = b_s.unsqueeze(0), b_a.unsqueeze(0)
        # planner.policy.learn_batch(b_s, b_a)
        ##################
        pred_traj = planner.plan(*obs)
        #

        vis.cla()
        vis.lazy_draw(*obs, traj)
        vis.draw_trajectory(pred_traj, '.-r',show_footprint=False)
        vis.pause(0.01)

