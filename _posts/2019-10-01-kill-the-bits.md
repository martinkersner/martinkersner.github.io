---
layout: post
title:  "Kill the bits and gain the speed?"
date:   2019-10-01 00:0:00 +0000
disqus_identifier: 2019-10-01
author: martin
comments: true
summary: Recently, Facebook AI Research in collaboration with University of Rennes released paper “And the Bit Goes Down&#58; Revisiting the Quantization of Neural Networks” which was submitted to ICLR 2020. The authors proposed a method of weight quantization for ResNet-like architectures using Product Quantization. Unlike many other papers, the error caused by codewords was not minimized directly. The training method aims to minimize the reconstruction error of fully-connected and convolutional layer activations using weighted k-means. Quantization was applied to all 3x3 and 1x1 kernel sizes except for the first convolutional layer. The paper emphasizes the importance of optimizing on in-domain input data in both quantizing and fine-tuning stages. Using their technique, weights in ResNet50 can be compressed with a 20x factor while maintaining competitive accuracy (76.1&nbsp;%). The potential impact of byte-aligned codebooks on efficient inference on CPU was briefly mentioned, but no actual method was presented. We propose and explore one possible way of exploiting frequent redundant codewords across input channels in order to accelerate inference on mobile devices.
---

Recently, Facebook AI Research in collaboration with University of Rennes released paper **“And the Bit Goes Down: Revisiting the Quantization of Neural Networks”** <a href="#kill-the-bits-ref">[1]</a> which was submitted to <a href="https://openreview.net/forum?id=rJehVyrKwH" target="_blank">ICLR 2020</a>.
The authors proposed a method of weight quantization for ResNet-like <a href="#resnet-ref">[2]</a> architectures using Product Quantization <a href="#product-quantization-ref">[3]</a>.
Unlike many other papers, the error caused by codewords was not minimized directly.
The training method aims to minimize the reconstruction error of fully-connected and convolutional layer activations using weighted $k$-means.
Quantization was applied to all 3x3 and 1x1 kernel sizes except for the first convolutional layer.
The paper emphasizes the importance of optimizing on in-domain input data in both quantizing and fine-tuning stages.
Using their technique, weights in ResNet50 can be compressed with a 20x factor while maintaining competitive accuracy (76.1&nbsp;%).
The potential impact of byte-aligned codebooks on efficient inference on CPU was briefly mentioned, but no actual method was presented.
We propose and explore one <a href="#inference-acceleration">possible way of exploiting frequent redundant codewords</a> across input channels in order to accelerate inference on mobile devices.

<!-- TODO BISONAI -->
<!-- TODO why mobile machine learning? -->

The following post is divided into two parts: <a href="#method-overview">Method Overview</a>, and <a href="#inference-acceleration">Inference Acceleration</a>.

<a class="anchor" id="method-overview"></a>
## Method Overview
We present an overview of methods used in a paper: <a href="#product-quantization">Product Quantization</a>, <a href="#codebook-generation">Codebook Generation</a> and <a href="#network-quantization">Network Quantization</a>.
For more details, we recommend reviewing the original paper <a href="#kill-the-bits-ref">[1]</a>.

<a class="anchor" id="product-quantization"></a>
### Product Quantization
Product Quantization (PQ) is a general quantization technique that enables a joint quantization of arbitrary number ($d$) of components.
The number of components is a hyperparameter, but as we can see later it can be chosen based on <a href="#conv-split">prior information about the data that we want to quantize</a>.
The dimensionality $d$ of quantized components does not change and neither does the data type precision.
The main benefit of PQ comes from its compressed representation of quantized components defined by indexes pointing to a **codebook**.
The codebook of dimensions $d \times k$ stores $k$ optimal centroids also called **codewords**.
The number of codewords $k$ directly affects the size of a codebook and the data type that is used to store codeword indexes is defined by the number of codewords in the codebook (e.g. 1 Byte can hold up to 256 indexes).
The codewords are derived from data, and the more frequent components can be quantized with a lower error.
The quantization process itself is usually done using one of the clustering techniques, such as <a href="https://en.wikipedia.org/wiki/K-means_clustering" target="_blank">weighted $k$-means</a> in the case of "And the Bit Goes Down"<a href="#kill-the-bits-ref">[1]</a>.

<center>
<img src="{{ "/img/blog/kill-the-bits/codebook.png" | prepend: site.url }}" width="90%">
</center>

<a class="anchor" id="codebook-generation"></a>
### Codebook Generation
Product Quantization was utilized to quantize the weights of convolutional and fully-connected layers.
The obvious objective of weight quantization is to minimize an error between original and quantized weights (Equation 1).
This can seem like a valid approach, however, we should keep in mind that eventually, weights are just constants that are used to compute the actual activations.
For this reason, an imprecise quantization does not manifest itself only in a *weight error* but also in a *reconstruction error* of the layer output.
On the other hand, this objective requires to have access only to weights, and not the training data.

<a class="anchor" id="eq-1"></a>

$$
\begin{align}
|| W - \hat{W} ||_{2}^{2} = || W - q(W) ||_{2}^{2}
\end{align}
$$

*Equation 1: Objective function of quantization method for minimizing weight error.*

The authors of the paper proposed an alternative objective function (Equation 2) that minimizes the reconstruction error.
This objective function can be applied only during training to have access to input activations $x$.
The <a href="#network-quantization">training procedure</a> is described in the following subsection.

<a class="anchor" id="eq-2"></a>

$$
\begin{align}
|| y - \hat{y} ||_{2}^{2} = || x (W - q(W)) ||_{2}^{2}
\end{align}
$$

*Equation 2: Objective function of quantization method for minimizing reconstruction error.*

The objective function from equation 2 can be minimized using weighted $k$-means in which weights are represented by input activations, and therefore the reconstruction error is scaled by the magnitude of input activations.

Until now, we have talked about layer weights and quantization in general but to fully exploit Product Quantization, layer weights should be split into coherent subvectors.
This split is different for various layer types.
Here, we describe the split for a fully-connected layer and later for a <a href="#conv-split">convolutional layer</a>.
The weight $W_{fc} \in R^{C_{in} \times C_{out}}$ of the fully-connected layer is first split into columns and every column is then further split into $m$ subvectors of size $d = C_{in} / m$, assuming that $C_{in}$ is divisible by $m$.
Generated subvectors are utilized for quantization and the generation of codewords within the codebook.

<center>
<img src="{{ "/img/blog/kill-the-bits/fc-split.png" | prepend: site.url }}" width="35%" >
</center>

*Figure 2: Visualization of weight split into subvectors in a fully-connected layer.*

<a class="anchor" id="conv-split"></a>
Convolutional weight $W_{conv} \in R^{C_{out} \times C_{in} \times K \times K}$ has an implicit spatial correlation between $K$'s dimensions where $K$ depicts the filter size.
Using this knowledge, weight is reshaped into $W_{conv} \in R^{(C_{in} \times K \times K) \times C_{out}}$ and split along the first dimension into subvectors of size $K^2$ (e.g. subvectors of size 9 for 3x3 convolution).


<a class="anchor" id="network-quantization"></a>
### Network Quantization
Network quantization starts from a pre-trained network and in its first phase layers are *quantized and finetuned* independently in a sequential manner, from the lowest ones up to the final output layer.
Input activations $x_{i}$ in layer $L_{i}$ are used in the process to quantize weights $W_{i}$ and according to <a href="#kill-the-bits-ref">[1]</a>, 100 finetuning steps for every layer are sufficient to converge.
During finetuning, the quantized network is optimized using <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence" target="_blank">KL&nbsp;loss</a> between the output of the quantized network and output from floating-point *teacher network* <a href="#distillation-ref">[4]</a> of the same architecture.

After all layers are quantized and locally finetuned, *global finetuning*, the second phase of network quantization, can start.
The global finetuning phase trains all codebooks jointly and additionally, a running mean and variance of Batch Normalization layers <a href="#batchnorm-ref">[5]</a> are updated as well.

<a class="anchor" id="inference-acceleration"></a>
## Inference Acceleration
“And the Bit Goes Down” <a href="#kill-the-bits-ref">[1]</a> achieves good compression ratios, using PQ and half-precision floating-point weights, however, the paper does not go beyond the compression use case.
Network compression is usually just one necessary ingredient for deploying trained networks on edge devices.
With smaller networks we can save the bandwidth while transmitting the latest model to the remote device, and also save the space in a local memory of a device, however, the *memory footprint* and *inference time* are no less important for edge devices.
In this section, we will describe one possible <a href="#acceleration-proposal">acceleration technique</a> for the convolutional layer in the PQ network and <a href="#benchmarking-optimized-convolution">evaluate</a> it on a mobile device.

<a class="anchor" id="acceleration-proposal"></a>
### Acceleration Proposal
One famous technique that compresses the size of a network and implicitly accelerates its inference time is *channel pruning* <a href="#channel-pruning-ref">[6]</a>.
Channel pruning removes whole channels from the given weight, which results in less computation in the current layer and also in the previous one.

Weights in PQ network have the same shape as those in an original non-quantized network, therefore we cannot achieve similar speedups as with channel pruning technique, however, we can exploit the fact that every layer has limited number of unique codewords and that some of those codewords can repeat within the same group of input channels.
In the figure below, we conceptualize repeated codewords within the same group of input channels using colors (every column has at least two identical codewords).

<center>
<img src="{{ "/img/blog/kill-the-bits/fc-split-repeat.png" | prepend: site.url }}" width="30%" >
</center>

*Figure 3: Visualization of weight split into subvectors in a fully-connected layer with repeated codewords within the same group of input channels.*

Convolution is composed of two operations: multiplication and addition, and for single output value can be defined as follows:

<a class="anchor" id="conv-equation-1"></a>

$$
\begin{align}
w_0 * x_0 + w_1 * x_1 + ... + w_{K^{2}-2} * x_{K^{2}-2} + w_{K^{2}-1} * x_{K^{2}-1},
\end{align}
$$

where $x_i$ represents single value input activation, $w_i$ is a single value from weight $W$, and $K^2$ defines weight size.
Convolution satisfies <a href="https://en.wikipedia.org/wiki/Convolution#Properties" target="_blank">distributive property</a> which we will use to decrease the number of multiplication operations.
In the following equation, we demonstrate how the optimized convolution would work.
Notice that the red color weight $\color{red}{w_0}$ repeats three times.
This allows us to first sum up activations that were paired with $\color{red}{w_0}$ and only after that we multiply $\color{red}{w_0}$ with summed activations.
With this approach, a number of saved multiplication operations is proportional to number of repeated weight values.

<a class="anchor" id="conv-equation-2"></a>

$$
\begin{align}
\color{red}{w_0} * \color{blue}{x_0} + \color{red}{w_0} * \color{green}{x_1} + ... + \color{red}{w_0} * \color{purple}{x_{K^{2}-2}} + w_{K^{2}-1} * x_{K^{2}-1} \\
\color{red}{w_0} * (\color{blue}{x_0} + \color{green}{x_1} + \color{purple}{x_{K^{2}-2}}) + ... + w_{K^{2}-1} * x_{K^{2}-1}
\end{align}
$$

This method can be readily applied to PQ networks because with a limited number of unique codewords we can expect a codeword redundancy within the same group of input channels.
In the next section, we describe how we implemented and evaluated the proposed acceleration technique for ResNet 18 network.

<a class="anchor" id="implementation-and-analysis"></a>
## Analysis & Implementation
The authors released an <a href="https://github.com/facebookresearch/kill-the-bits" target="_blank">implementation of paper</a> together with several <a href="https://github.com/facebookresearch/kill-the-bits/tree/master/src/models/compressed" target="_blank">pre-trained models</a>.
For our analysis, we use ResNet 18 network as an example and all measurements are taken on <a href="https://www.gsmarena.com/oneplus_6t-9350.php" target="_blank">One Plus 6t</a> running modified <a href="https://github.com/bisonai/ncnn" target="_blank">ncnn</a> inference engine.

### Theoretical Speedup
The computational cost of standard convolution is defined as

$$
\begin{align}
K \times K \times M \times N \times D_x \times D_x,
\end{align}
$$

where $K \times K$ represents kernel size, $M$ number of input channels, $N$ number of output channels, and $D_x \times D_x$ size of input activation.
<a href="#acceleration-proposal">Proposed acceleration</a> can be viewed as a preprocessing step of input activations before they are fed into a convolution operation.
Such preprocessing would channel-wise sum up input activations that share the same weight codewords and as a result number of input channels would decrease.

To be able to compute a theoretical speedup of proposed acceleration we must have access to the codebook for every layer.
Fortunately, the pre-trained ResNet 18 model contains such <a href="https://github.com/facebookresearch/kill-the-bits/blob/130a56c028de3b6af5cdee397ddb9d220246b7cb/src/inference.py#L71-L72" target="_blank">codeword indexes</a>.
From this information, we can obtain a number of unique codewords that would be used for the computation of a single output channel.
The number of unique codewords varies between output channels, however, to simplify the computation of the theoretical speedup, we decided to make use of the maximum number of unique codewords across all output channels for every layer.
With this, we obtain a **minimal theoretical speedup reaching almost 20&nbsp;% of computational cost**.
In the figure below, you can see that the reduction of input channels is more pronounced in the latter parts of the network, where the original number of input channels stretches up to 512.

<center>
<img src="{{ "/img/blog/kill-the-bits/channel-reduction.png" | prepend: site.url }}" width="80%">
</center>

*Figure 4: Number of input channels before and after preprocessing of input activations for every convolutional layer.*

### Layer-wise Analysis
To put a theoretical speedup into perspective, we measured an inference time of floating-point ResNet 18 network.
The execution of all layers took about 187&nbsp;ms.
From the figure below, we can see that most of the time (89.2&nbsp;%) is spent in 3x3 convolutions and the second most time-consuming layer is 7x7 convolution.
Our focus is only on 3x3 convolution because 7x7 convolution is part of the first layer which was not quantized.

<center>
<img src="{{ "/img/blog/kill-the-bits/breakdown-resnet18.png" | prepend: site.url }}" width="70%">
</center>

*Figure 5: Layer type-wise inference time breakdown of floating-point Resnet 18 network executed on Snapdragon 845.*

Many 3x3 convolutional layers cost around 11&nbsp;ms.
The most frequent input activation shapes are 56x56, 28x28, 14x14 and 7x7 with 64, 128, 256 and 512 input channels, respectively.
Later, we will <a href="#benchmarking-optimized-convolution">benchmark</a> these convolutions to find out the actual computational speedup.

### Implementation
To be able to verify the theoretical speedup, we <a href="https://github.com/bisonai/ncnn" target="_blank">modified ncnn</a> inference engine and integrated our proposed inference acceleration method.
We added a <a href="TODO" target="_blank">channel-wise summation of input activations</a> with a randomly generated codeword assignments limited by the number of unique weight codewords per input channel.
The summation was implemented using <a href="https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics" target="_target">NEON intrinsics</a> to exploit parallel processing capabilities of the ARM processor and the memory used to store the summed up input activations were allocated in advance, during the network initialization.
Convolutional weights need to be altered in a way that acknowledges changes within input activations, specifically by shifting and removing weight codewords.
In our implementation, however, we allocate only weights of correct shape, without correct weight initialization.
While this implementation promises an accelerated inference time, it should be noted that it has a larger memory footprint due to the extra preallocated input activations and weights.

<a class="anchor" id="benchmarking-optimized-convolution"></a>
### Benchmarking Optimized Convolution
In the following figures, we compare the inference time of original 3x3 convolution (denoted as *w/o reduction* and displayed in red color) with 3x3 convolution enhanced by preprocessing of input activation (denoted as *w/ reduction* and displayed in blue color).

<center>
<img src="{{ "/img/blog/kill-the-bits/speedup_7x7.png" | prepend: site.url }}" width="70%">
</center>

*Figure 6: Comparison of 3x3 convolution with and without channel reduction for 7x7 input.*

<center>
<img src="{{ "/img/blog/kill-the-bits/speedup_14x14.png" | prepend: site.url }}" width="70%">
</center>

*Figure 7: Comparison of 3x3 convolution with and without channel reduction for 14x14 input.*

<center>
<img src="{{ "/img/blog/kill-the-bits/speedup_28x28.png" | prepend: site.url }}" width="70%">
</center>

*Figure 8: Comparison of 3x3 convolution with and without channel reduction for 28x28 input.*

We notice that the inference time linearly increases with the number of input channels.
We can also see that to gain any speed, the number of channels has to be reduced approximately by a 4x factor.
The overhead in channel reduction seems to be quite high.
It is caused by access to every channel of input activation and channel-wise summation operation.
Moreover, the implementation of the convolutional layer exploits <a href="http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0489i/CIHBGIGD.html" target="_blank">`vmla` SIMD instruction</a> that can multiply and accumulate four single-precision floating-point values in parallel using only one instruction.

```c++
float32x4_t vmlaq_f32 (float32x4_t a, float32x4_t b, float32x4_t c)
```

*Code 1: `vmlaq_f32` ARM Neon instruction can perform 4 multiplications of 32-bit floating-point and 4 additions of 32-bit floating-point in parallel using one instruction.*


As a result, the summation in the convolutional layer is used for free.
Our method, therefore, does not move it out of convolution as it might seem from <a href="#conv-equation-2">equation&nbsp;2</a> but introduces extra new summation operations.
Unfortunately, with the current pre-trained PQ ResNet 18 network, where the largest channel compression per layer is slightly above 50&nbsp;%, we wouldn't be able to accelerate the inference, unless the number unique codewords for every channel gets smaller.

<a class="anchor" id="summary"></a>
## Summary
**“And the Bit Goes Down: Revisiting the Quantization of Neural Networks”** <a href="#kill-the-bits-ref">[1]</a> proposes a quantization technique that combines a Product Quantization with careful layer-wise pretraining and local/global finetuning using fixed teacher network.
Network weights are quantized using weighted $k$-means with an objective function that tries to minimize a reconstruction error, instead of minimizing weight error directly.
The proposed method achieves high compression rates on Resnet-like architectures, however, there was no suggestion in the paper how we could accelerate an inference time with such a quantization scheme.

We proposed a method that modifies input activations and convolutional weights to reduce the number of multiplications in a convolutional layer.
By benchmarking on a One Plus 6t mobile device, we confirmed a speedup of convolution using our method.
However, in order to achieve any meaningful speed gains the number of channels has to be reduced by at least a factor of 4x.

<a class="anchor" id="notes"></a>
## Notes
All experiments were launched using a <a href="TODO" target="_blank">modified ncnn benchmark</a> on a One Plus 6t with Snapdragon 845.
Every measurement was obtained by averaging 200 continuous runs and removing outliers from the first and last quartile of inference duration distribution.
Lastly, we always used only a single big core, since it reflects the best an allowed computational power in real-world scenarios.

<a class="anchor" id="references"></a>
## References
<a id="kill-the-bits-ref"></a>
[1] P. Stock, A. Joulin, R. Gribonval, B. Graham, H. Jégou: And the Bit Goes Down: Revisiting the Quantization of Neural Networks, 2019, <a href="https://arxiv.org/abs/1907.05686" target="_blank">link</a>
<br/>
<a id="resnet-ref"></a>
[2] K. He, X. Zhang, S. Ren, J. Sun: Deep Residual Learning for Image Recognition, 2015, <a href="https://arxiv.org/abs/1512.03385" target="_blank">link</a>
<br/>
<a id="product-quantization-ref"></a>
[3] H. Jegou, M. Douze, C. Schmid: Product Quantization for Nearest Neighbor Search, 2011, <a href="https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf" target="_blank">link</a>
<br/>
<a id="distillation-ref"></a>
[4] G. Hinton, O. Vinyals, J. Dean: Distilling the Knowledge in a Neural Network, 2015, <a href="https://arxiv.org/abs/1503.02531" target="_blank">link</a>
<br/>
<a id="batchnorm-ref"></a>
[5] S. Ioffe, C. Szegedy: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, 2015, <a href="https://arxiv.org/abs/1502.03167" target="_blank">link</a>
<br/>
<a id="channel-pruning-ref"></a>
[6] Y. He, X. Zhang, J. Sun: Channel Pruning for Accelerating Very Deep Neural Networks, 2017, <a href="https://arxiv.org/abs/1707.06168" target="_blank">link</a>
<br/>
