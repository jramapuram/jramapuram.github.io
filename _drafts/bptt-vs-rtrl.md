---
layout: post
title: BPTT vs. RTRL
categories: ramblings
---

RNN's have become immensely popular for a variety of tasks involving sequence modelling ranging from speech recognition to video analysis and many others. One aspect that is not discussed very much are the consequences of the training method employed. While there are a plethora of non-gradient methods such as simulated annealing and particle swarm optimization we will focus our efforts on the two popular gradient based methods: RTRL (real-time recurrent learning)  & BPTT (backprop through time). We will also be focusing on the LSTM since it is the most popular RNN architecture employed.

# BPTT
Backprop through time has become the most popular method of training. A pro/con list is shown below

| Pros                                                | Cons                                    |
|-----------------------------------------------------|-----------------------------------------|
|Uses Exact Gradient                                  | Works on fixed window (truncated BPTT)  |
|Easy to verify via numerical gradient                | Not fully online                        |
|Possibility to learn outside of truncated window (*) | Higher memory cost than RTRL            |

(*) has been shown to be true (at least for short time intervals). Something that has yet to evaluated is whether an LSTM trained using BPTT can truly remember information it learnt for a large time interval.

A great numerical example of BPTT (as well as the equations below) can be worked through at the [blog of Aidan Gomez](https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/):

## Forward Pass:

$$
\begin{bmatrix} \hat{a}_{t}\\ \hat{i}_{t}\\ \hat{f}_{t}\\ \hat{o}_{t} \end{bmatrix}\
 = \begin{bmatrix} W_{a}\\ W_{i}\\ W_{f}\\ W_{o} \end{bmatrix} x_t\
 + \begin{bmatrix} U_{a}\\ U_{i}\\ U_{f}\\ U_{o} \end{bmatrix} out_{t-1}\
 +  \begin{bmatrix} b_{a}\\ b_{i}\\ b_{f}\\ b_{o} \end{bmatrix}
$$

Applying the non-linearities & getting the state and output evaluations at the current time:

$$
\begin{bmatrix} a_{t}\\ i_{t}\\ f_{t}\\ o_{t} \end{bmatrix}\
= \begin{bmatrix} \gamma(\hat{a}_{t})\\ \sigma(\hat{i}_{t})\\ \sigma(\hat{f}_{t})\\ \sigma(\hat{o}_{t}) \end{bmatrix}\
$$

$$
state_{t} = a_{t} \odot i_{t} + f_{t} \odot state_{t-1} \\
out_{t} = \gamma(state_{t}) \odot o_{t}
$$

## Backward Pass:

$$
\begin{aligned}
  \delta out_{t} &= \Delta_{t} + \Delta out_{t} \\
  \delta state_{t} &= \delta out_{t} \odot o_{t} \odot \gamma'(state_{t}) + \delta state_{t+1} \odot f_{t+1}\\
  \delta a_{t} &= \delta state_{t} \odot i_{t} \odot (1 - a_{t}^{2})\\
  \delta i_{t} &= \delta state_{t} \odot a_{t} \odot i_{t} \odot (1 - i_{t})\\
  \delta f_{t} &= \delta state_{t} \odot state_{t-1} \odot f_{t} \odot (1 - f_{t})\\
  \delta o_{t} &= \delta out_{t} \odot \gamma(state_{t}) \odot o_{t} \odot (1 - o_{t})\\
  \delta x_{t} &= W^{T} \cdot \delta gates_{t}\\
  \Delta out_{t-1} &= U^{T} \cdot \delta gates_{t}
\end{aligned}
$$

### Working out the chain rule for the LSTM
As a small aside I want to work through the logic on how to derive the above equations via the chain rule. This can actually also be done via a langrangian formulation as well which can be seen [here](http://www.argmin.net/2016/05/18/mates-of-costate/) for the standard ANN case.



# RTRL
Real-time recurrent learning was the initial method proposed by Williams and Zipser in 1989 to train RNN's. The problem with vanilla RTRL is that it is $$O(n^4)$$ in computational complexity! You can read more about why this would be the case here on [Colah's Blog](http://colah.github.io/posts/2015-08-Backprop/). Thankfully however the case we are going to examine is the LSTM with internal RTRL (proposed by Schmidhuber & Felix Gers) has the same computational complexity as BPTT. The key idea is here is that the internals of the cell will update their parameters
