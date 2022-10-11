#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 21:52:01 2021

@author: Alina Bialkowski
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#======================
# 9.1 (a)
A1 = [2.66,3.21,1.87,1.69]
A2 = [1.25, 2.34]

# compute the mean of each action
u1 = np.mean(A1)
u2 = np.mean(A2)

print(f'Mean rewards:\nA1: {u1}\nA2: {u2}')

#======================
# 9.1 (b)
conf_val1 = np.sqrt((5*np.log(6))/4.0)
conf_val2 = np.sqrt((5*np.log(6))/2.0)

ucb1 = u1 + conf_val1
ucb2 = u2 + conf_val2

print(f'UCB:\nA1: {ucb1}\nA2: {ucb2}')

#======================
# Make a bar chart of UCB values for 9.1(b)
# set width of the bars
width = 0.35       

# Copy the data to lists
labels = ['A1', 'A2']
mean_vals = [u1, u2]
conf_ints = [conf_val1, conf_val2]
ucb_vals = [ucb1, ucb2]

# Make plot
fig,ax = plt.subplots()
ax.bar(labels, mean_vals, width, label='Mean Value')
ax.bar(labels, conf_ints, width, bottom=mean_vals, label='Confidence Interval')
#ax.grid(True, axis='y')
ax.set_xlabel('Action')
ax.set_ylabel('UCB value')
ax.set_ylim([0, 5.5])
ax.legend(loc='upper center')
ax.axes.set_xlim(-0.5,1.5)

# Add labels to the bars
for i, v in enumerate(ucb_vals):
    ax.text(i-0.065, v+0.1, '{:.2f}'.format(v)) #, color='black', fontweight='bold')
    
plt.savefig('UCB_values.pdf')

#======================
# Exercise 9.1 (c)
# Plot the distributions of rewards for each arm from samples
import MAB_arms as ar
import MAB_instance
import MAB_policies

# Weibull
a = 2. #shape param
b = 2*np.sqrt(2) #scale param
weibull_arm = ar.WeibullArm(shape=a, scale=b)

# Normal
mean_val = 3
std_val = 1
normal_arm = ar.NormalArm(mu=mean_val, sigma=std_val)

def draw_dist(arm, label=None, color=None):
    # Draw samples from the arm
    samples = [arm.draw() for _ in range(10000)]
    if color is not None:
        sns.distplot(samples,label=label, color=color)
    else:
        sns.distplot(samples,label=label)
    plt.xlabel('Reward')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    print(f'{label} sample mean = {np.mean(samples)}')
    
fig,ax = plt.subplots()
draw_dist(normal_arm, label='Normal (A1)', color='violet')
draw_dist(weibull_arm, label='Weibull (A2)', color='cornflowerblue')
plt.savefig('Normal_Weibull.pdf')

#======================
# Compute the probability distributions directly from equations
# set the x-values to compute the pdfs for
x = np.linspace(0,8,400)

# Weibull pdf (probability density function)
def weib(x,n,a):
    return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)
# Normal pdf
from scipy.stats import norm

# # Draw samples from the 2 arms
# samples_normal = [normal_arm.draw() for _ in range(10000)]
# samples_weibull = [weibull_arm.draw() for _ in range(10000)]
# # Compute histogram & plot the pdf on top
# #results = np.random.weibull(5.,1000))
# count, bins, ignored = plt.hist(samples_weibull)
# scale_to_max_bin = count.max() / weib(x, b, a).max()
# plt.plot(x, weib(x, b, a)*scale_to_max_bin) #
# plt.show()

# count, bins, ignored = plt.hist(samples_normal)
# scale_to_max_bin = count.max() / norm.pdf(x, mean_val, std_val).max()
# plt.plot(x, norm.pdf(x, mean_val, std_val)*scale_to_max_bin)
# plt.show()

# Plot the PDFs directly from analytical equations
fig,ax = plt.subplots()
ax.plot(x, norm.pdf(x, mean_val, std_val), label='Normal dist')
ax.plot(x, weib(x, b, a), label='Weibull dist')
ax.set_xlabel('Reward')
ax.set_ylabel('Probability')
ax.set_title('Weibull & Normal distributions')
fig.savefig('Normal_Weibull_analyticalPDFs.pdf')

#============
# Exercise 9.1 (d) Compare policies
arms = [weibull_arm, normal_arm]
policies = [MAB_policies.EpsilonGreedy(0.1), MAB_policies.UCB1()]
fig,ax = plt.subplots()
def test_policy(arms_, policy, n):
    trials, rewards = MAB_instance.Instance(arms_, policy).sim_policy(n)
    sns.distplot(rewards)
    print(f'Results for Policy {policy.name}'.center(50, "-"))
    print(f'Accumulated rewards = {np.sum(rewards)}')
    counts = np.bincount(trials.astype('uint8'))
    print('Trials spent on arms:')
    for idx in range(len(arms)):
        print(f"Machine {idx}: {counts[idx]} ({round(counts[idx]/n * 100)}%)")
    
    return trials,rewards
        
# Here, it looks like Eps policy is better
N = 100
trials_e,rewards_e = test_policy(arms, policies[0], N)
trials_ucb,rewards_ucb = test_policy(arms, policies[1], N)
plt.xlabel('Reward')
plt.ylabel('Probability')
plt.title(f'Comparing Epsilon Greedy & UCB1 policies for {N} trials')

print(f'Eps-greedy = {np.mean(rewards_e)} vs UCB = {np.mean(rewards_ucb)}')
plt.savefig('ComparePolicies.pdf')


#=============================================================
# ==== Extra analysis of epsilon greedy eps value for fun ====
# Create 5 arms with mean rewards as follows
# out of 5 arms, 1 arm is clearly the best
means = [0.1, 0.1, 0.1, 0.1, 0.9]
arms = []
for mean in means:
    arms.append(ar.NormalArm(mu=mean, sigma=1))

# Create simulations for each exploration epsilon value
fig,ax = plt.subplots()
for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]:
    algo = MAB_policies.EpsilonGreedy(epsilon)
    results = test_policy(arms, algo, 5000)
    
    ax.plot(np.cumsum(results[1]), label=f'$\epsilon$={epsilon}')
    
# Run UCB for comparison
algo = MAB_policies.UCB1()
results = test_policy(arms, algo, 5000)
ax.plot(np.cumsum(results[1]), label='UCB1', color='black')

ax.set_ylabel('Cumulative reward')
ax.set_xlabel('Trial number = total number of #arm pulls')
ax.set_title('Comparing Epsilon Greedy policy with different $\epsilon$ values')

ax.legend()
fig.savefig('5Arms_DiffPolicy_cumulative_rewards.pdf')
