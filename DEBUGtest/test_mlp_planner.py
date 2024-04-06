import torch.nn.functional

from spider.data.Dataset import OfflineLogDataset
from spider.planner_zoo.MlpPlanner import MlpPlanner

from torch.utils.data.dataloader import DataLoader


# setup the planner
planner = MlpPlanner({
    "steps": 20,
    "dt": 0.2,
})

# setup the dataset
dataset = OfflineLogDataset('./dataset/', planner.state_encoder, planner.action_encoder)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# train_mode the planner
import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

criterion = torch.nn.MSELoss()  # torch.nn.L1Loss()
optimizer = optim.Adam(planner.policy.parameters(), lr=0.0001)

planner.train_mode()
avg_losses = []
for epoch in tqdm.tqdm(range(100)):
    temp_losses = []
    for i, exp in enumerate(dataloader):
        states, actions = exp[:2]
        states = states.to(planner.device)
        actions = actions.to(planner.device)

        pred_actions = planner.policy(states)
        loss = criterion(actions, pred_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        temp_losses.append(loss.item())

    avg_losses.append(sum(temp_losses)/len(temp_losses))
    plt.cla()
    plt.plot(avg_losses)
    plt.pause(0.001)
plt.savefig("mlp_train.png")
plt.close()

# save the model
torch.save(planner.policy.state_dict(), 'mlp.pth')

# load the model
planner.policy.load_state_dict(torch.load('mlp.pth'))

# test the planner
from spider.interface.BaseBenchmark import DummyBenchmark
benchmark = DummyBenchmark({
})
benchmark.test(planner)

