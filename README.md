# Homework 3: Policy Gradient Implementation
This is HW3 in my course "Introduction of deep learning and reinforcement learning".

## Assignment

In this homework, I provide a generic RL framework with which you can implement your own RL algorithms (on TensorFlow). Your task is to implement the Policy Gradient algorithm I introduced in class.

That is, you need to fill in the blank in the `_train(...)` method defined in file **<./algo/pg.py>**.

## Getting Started

### Prerequisites

Only Python 3.6+ is supported.

To install dependencies, run:

```
pip install -r requirements.txt
```

## Examples

To train policies in [gym](http://gym.openai.com/envs/) classic control environments, run:

```
python run.py --env=CartPole --n_itr=100 --viz
```

- `env` specifies the environment name. (This flag is optional. Available environments include 'CartPole', 'Acrobot', and 'MountainCar'. 'CartPole' by default.)
- `n_itr` specifies the number of training iterations. (This flag is optional. 100 by default.)
- `viz` If this flag exists, visualize the performance of policy in training process. (This flag is optional. You can remove this flag if you don't want to visualize them.)
