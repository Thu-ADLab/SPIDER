from spider.data.DataCollector import DataCollector
from spider.planner_zoo import LatticePlanner
from spider.interface import DummyInterface

collector = DataCollector(
    LatticePlanner({
        "steps": 20,
        "dt": 0.2,
        "print_info": False
    }),
    DummyInterface(),
)

collector.collect(100)