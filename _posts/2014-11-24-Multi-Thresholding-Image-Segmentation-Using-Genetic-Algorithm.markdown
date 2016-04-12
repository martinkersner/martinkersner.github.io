---
layout: post
comments: true
title:  "Multi-Thresholding Image Segmentation Using Genetic Algorithm"
date:   2014-11-24 10:10:00
disqus_identifier: 20141124
categories: ['Genetic algorithm', 'Image segmentation', 'Multi-thresholding']
---

Paper ([Multi-Thresholding Image Segmentation Using Genetic Algorithm][paper-pdf]) proposes a method of image segmentation with multiple thresholds, which are determined by genetic algorithm. Multi-thresholding allows to segment more objects contained in image, however it could be extremely time consuming to find out these thresholds.

### Basic terminology of genetic algorithms

Typically, population size represent the number of solutions per iteration.
Chromosome is in other words solution.
Genes (most often represented as bits) form a chromosome.

### Genetic algorithm has four stages

1. Initialization,
2. Evaluation of fitness,
3. Reproduction,
    * Selection ([roulette wheel][roulette-wheel]),
    * Crossover,
    * Mutation,
    * Accepting the solution.
4. Termination.

### Notes

Firstly they convert image to level of grays to reduce posterior computing time.

They represent chromosomes as bit vectors of length *L * n*, where *L* denotes *log(number of gray levels)* and *n* is the number of desired thresholds.

The fitness function is computed as ratio between inter-object variance and intra-object variance. Lower is better.

At the final step of reproduction they compare created descendants with their parents and accept to new population only these which perform better.

Genetic algorithm is terminated after particular number of iterations.

[paper-pdf]: http://www.worldcomp-proceedings.com/proc/p2011/IPC8346.pdf
[roulette-wheel]: https://en.wikipedia.org/wiki/Fitness_proportionate_selection
