# CrowdGAN
Pytorch implementation for the paper:

[**Data-driven Crowd Simulation with Generative Adversarial Networks**](https://dl.acm.org/doi/abs/10.1145/3328756.3328769)  
Authors: *<a href="http://people.rennes.inria.fr/Javad.Amirian/">Javad Amirian</a>,
<a href="https://team.inria.fr/rainbow/wouter-van-toll/">Wouter van-Toll</a>,
<a href="http://aplicaciones.cimat.mx/Personal/jbhayet">Jean-Bernard Hayet</a>,
<a href="http://people.rennes.inria.fr/Julien.Pettre/">Julien Pettre</a>*  
Presented at [CASA 2019](https://casa2019.sciencesconf.org/) (Computer Animation and Social Agents)  [[arxiv](https://arxiv.org/pdf/1905.09661.pdf)], [[slides]()] 

## System Overview
Generally a GAN system is composed of a Generator and a Discriminator.
On the left side of the figure below, you see the Trajectory Generator and on the right side, you see the Trajectory Discriminator. 
<p align='center'>
  <img src='figs/block-diagram.png' width='800px'\>
</p>

However for implementation we use two separate GANs:
1. **Entry-Point GAN**: which is responsible just for generating the first two points of a trajectory
2. **PredictorGAN**: which takes the beginning of a trajectory and predicts the rest of it step-by-step.

## Results
We tested our system on [ETH walking pedestrians dataset](https://vision.ee.ethz.ch/en/datasets/):

<p align='center'>
  <img src='figs/fake.gif' width='400px'\>
  <img src='figs/real.gif' width='400px'\>
</p>

## Training
For training of the system, you need to run entrypointGAN.py and predictorGAN.py separately.

### Hyper-parameters  
All the hyper-parameters are stored in [config.yaml](./config/config.yaml)
 

## Reference
If you use this code for your research, please cite our paper:
```
@inproceedings{amirian2019crowdgan,
  title={Data-Driven Crowd Simulation with Generative Adversarial Networks},
  author={Amirian, Javad and Van Toll, Wouter and Hayet, Jean-Bernard and Pettr{\'e}, Julien},
  booktitle={Proceedings of the 32nd International Conference on Computer Animation and Social Agents},
  pages={7--10},
  year={2019}
}
```
