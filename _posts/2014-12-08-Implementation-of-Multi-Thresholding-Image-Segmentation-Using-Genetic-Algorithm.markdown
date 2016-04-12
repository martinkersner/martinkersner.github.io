---
layout: post
comments: true
title:  "Implementation of Multi-Thresholding Image Segmentation Using Genetic Algorithm"
date:   2014-12-08 10:10:00
disqus_identifier: 20141208
categories: ['Genetic algorithm', 'Image segmentation', 'Multi-thresholding']
---

Project is inspired by [Multi-Thresholding Image Segmentation Using Genetic Algorithm][paper-pdf] paper. Image segmentation can be pursued by many different ways. One of them is called multi-thresholding. Since we want to segment image to more than two segments (more than one threshold) we need to determine at least two thresholds. If we want to segment gray level image with 256 levels to three segments we are likely supposed to examine 256 * 255 = 65,280 different threshold combinations. This brute force approach would end up with optimal solution,
however computing time would be definitely high. Genetic algorithm searches space containing all possible solutions and obtain the best solution among all examined in much less time than brute force algorithm.

<div style="text-align:center">
<img src="http://i.imgur.com/eo13WyX.png?1" />
</div>

The result of segmentation by genetic algorithm with population size 20 and number of iterations 30.

### Implementation

Genetic algorithm was implemented in GNU Octave. This language was selected due to its fast prototyping. The speed of algorithm is not the matter of interest.

#### Initialization

This part of code can be found at `ga_segmentation.m`. There are few settings which affect the result of algorithm.

* *n_population*: size of population; contains different solutions
* *n_iterations*: number of iterations; algorithm terminates after all iterations are done
* *n_thresholds*: number of desired thresholds; *n_thresholds* = (number of segments - 1)

The next three settings are related to each other. They are ratios of selection (*n_selection*), crossover (*n_crossover*) and mutation (*n_mutation*). The sum of all these parameters has to be equal to 1. This condition guarantees the size of population does not change between iterations.

Single chromose is implemented as vector of binary numbers. The length of vector is *L* * *n*, where *L* denotes *log(number of gray levels)* and *n* is the number of desired thresholds. Population is represented as a matrix, where each row is single chromosome and number of rows corresponds to size of population. An initial population was randomly generated.

In this part we also load an image for segmentation. Algorithm works only with grayscale images, so we always convert each image at the beginning before we start to work with it.

#### Evaluation of fitness

According to paper, which was mentioned above, evaluation of fitness should be performed with ratio between between inter-object variance and intra-object variance. We have decided for sligthly easier evaluation based on sum of intra-object variance. The solution with the lowest sum is the most accurate.

`fitness.m`, `fitness_one.m`


#### Selection

Current solution selects to new generation only the best solutions. The number of forwarded solution depends on ratio *n_selection*.

`first_best.m`

#### Crossover

Chromosomes, which are supposed to crossover, are chosen from two vectors containing radomly permutated indexes (numbers of row) of chromosomes. The number chromosomes, which crossover, depends on ratio *n_crossover*. One-point crossover is employed in current solution. The point of crossover is randomly generated with uniform distribution.

`crossover.m`, `crossover_one.m`

#### Mutation

Indexes (number of rows) of chromosomes are randomly permutated and the very first of them are more likely to be chosen. The number of chromosomes, which happens to be chosen for mutation, depends on ratio *n_mutation*. Current solution allows to mutate only one gene of chromosome.

`mutation.m`, `mutate_one.m`

### Experiments

Two experiments were performed:
The first one with picture composed from solid colors
and the second one uses benchmark from website [mosaic.utia.cas.cz][mosaic].
During the experiment we observed that 20 as size of population and 50 as number of iterations are satisfying for most of cases.

#### Solid colors

Though segmentation of this picture looks promising, we were not able to successfully segment all parts. The best result can be shown at the end of this subsection. We also noticed that algorithm is prone to insignificant segments (edges of shapes) with small area.

<img src="https://cloud.githubusercontent.com/assets/2312761/5239175/a63f604c-78d7-11e4-99a8-91ca7becaa40.png">

Source image contains 8 different colors.

<img src="https://cloud.githubusercontent.com/assets/2312761/5239187/b675129e-78d8-11e4-8a5d-faf4898c4997.png">
<img src="https://cloud.githubusercontent.com/assets/2312761/5239188/ba1d9b14-78d8-11e4-8fcc-556cc89a3db5.png">
<img src="https://cloud.githubusercontent.com/assets/2312761/5239189/bc544df6-78d8-11e4-9320-ef1d5003ffff.png">

2 thresholds, 3 segments

<img src="https://cloud.githubusercontent.com/assets/2312761/5239265/5de327b6-78dc-11e4-91af-753c98480033.png">

10 thresholds, 11 segments

#### Benchmark

Benchmark consist of 20 different images with artificially created texture segments. The number of segments is between 3 and 12, which can be sometimes even challenging for human.

Each triplet of images below represents source image, ground truth of segmentation and segmentation done by genetic algorithm. As we could expect the results have not reached high. Correct segmentation is 2.17 % overall.

<img src="https://cloud.githubusercontent.com/assets/2312761/5239296/5a7cfdd0-78dd-11e4-92a7-46d6ccd706e3.png"> 
<img src="https://cloud.githubusercontent.com/assets/2312761/5239297/5d1f5b50-78dd-11e4-9809-e6a5e00795ef.png">
<img src="https://cloud.githubusercontent.com/assets/2312761/5239299/6187ee32-78dd-11e4-9a17-fdd317b0209a.png">

2 thresholds, 3 segments

<img src="https://cloud.githubusercontent.com/assets/2312761/5239300/68d21fd2-78dd-11e4-9f0c-c32e04bac803.png">
<img src="https://cloud.githubusercontent.com/assets/2312761/5239301/6c8d395e-78dd-11e4-90ea-b68ad5035ecc.png">
<img src="https://cloud.githubusercontent.com/assets/2312761/5239305/70767f8a-78dd-11e4-9d89-138f1fc80731.png">

11 thresholds, 12 segments

### Conclusion

Genetic algorithm is able to find suboptiomal solution of multi-threshold segmentation. This solution is not suitable for all kinds of segmentation problems; e.g. binary segmentation: there are more reliable methods like Balanced histogram thresholding or Otsu's method. The low performance of texture segmentation was confirmed by benchmark.

The shortcoming of implemented solution is propensity to segment small areas like edges. These areas are not essential in segmentation and therefore should not be taken into account.

The algorithm could be refined by two-point crossover or roulette wheel for selecting chromosomes to the next population.

[paper-pdf]: http://www.worldcomp-proceedings.com/proc/p2011/IPC8346.pdf
[mosaic]: http://mosaic.utia.cas.cz
