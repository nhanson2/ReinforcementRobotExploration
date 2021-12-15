# DDPG_MountainCar


The mountain car continuous problem from gym was solved using DDPG, with neural networks
as function aproximators.
The solution is inspired in the DDPG algorithm, but using only low level information as inputs to the net, basically the net uses the position and velocity from the gym environment.
The exploration is done by adding Ornstein-Uhlenbeck Noise to the process. 


## Requirements:

- Numpy
- Tensorflow
- Open AI Gym


## How to run

Simply run

```
python mountain.py
```

If there is a model saved in the folder it will load and start the training/testing.