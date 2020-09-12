## Reinforcement Learning of Optimised Task Mappings

This repository contains prototype implementations for my work on learning optimised task-mappings via reinforcement learning. Here, I aim to leverage [Graph Convolutional Networks](https://tkipf.github.io/graph-convolutional-networks/) to learn mapping strategies that incorporate awareness of the placements and features of the other related tasks that scheduled on the same hardware. I therefore use code from Kipf and Welling's GCN repository [here](https://github.com/tkipf/gcn).

The repository is split into two implementation folers. The `src_simple_mlp` folder provides a simple implementation that uses the REINFORCE (and its baseline adaptation) RL algorithm to train a Multi-Layer Perceptron. The `src_ppo_gcn` folder provides a more complex implementation that uses [Proximal Policy Optimisation](https://arxiv.org/abs/1707.06347) to train a GCN model in an Actor-Critic setting, and also incorporates [Random Network Distillation](https://arxiv.org/abs/1810.12894) to encourage exploration.

**The prototype code as well as this README represents work-in-progress.**
