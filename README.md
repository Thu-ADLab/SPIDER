SPIDER -- Self-driving Planning and Intelligent Decision-making Engine with Reinforcement learning

The python package has been released to pypi: https://pypi.org/project/spider-python/

You are welcome to install it with `pip install spider-python` to have a try 
and provide valuable suggestions for further development of this project.

To have a glimpse of its capability, you can try to launch a demo script with 
```python
import spider
spider.teaser()
```

And what's more, if you have already got highway-env in your environment 
which can be installed with `pip install highway-env`, you can try another teaser 
about how spider gets access to the data interface conveniently and makes it easy to 
configure the environment:
```python
from spider.interface.highway_env import HighwayEnvBenchmarkGUI
HighwayEnvBenchmarkGUI.launch()
```

If you have any problem, please feel free to contact me:

Author: Zelin Qian(钱泽林)

Institution: School of Vehicle and Mobility, Tsinghua University, China

Email: qzl22@mails.tsinghua.edu.cn