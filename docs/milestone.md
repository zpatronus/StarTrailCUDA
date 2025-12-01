# **StarTrailCUDA: GPU-Accelerated Rendering of Night-Sky Star Trails Video** - Milestone Report

**Team Members:** Zijun Yang <zijuny@andrew.cmu.edu> and Jiache Zhang <jiachez@andrew.cmu.edu>

**Project Home Page:** [https://blog.zjyang.dev/StarTrailCUDA/](https://blog.zjyang.dev/StarTrailCUDA/)

![image-20251117192727072](./project_proposal.assets/image-20251117192727072.png)

<center><a href="https://www.bilibili.com/video/BV1Q64y1a7FE/?share_source=copy_web&vd_source=248bf19a901960bb7bbfb1705c664b9c&t=79">https://www.bilibili.com/video/BV1Q64y1a7FE/?share_source=copy_web&vd_source=248bf19a901960bb7bbfb1705c664b9c&t=79</a></center>

## LIST OF WORK COMPLETED SO FAR

exploring different algorithms:

- max (zjc)
- average (zjc)
- exponential (zjc)
  - hard to control tail length (zjc)
- linear (zjc)
- LINEARAPPROX: Uses mathematical heuristics to approximate the LINEAR algorithm with $O(1)$ cost instead of $O(window size)$ to render each frame. (Zijun Yang)

We explored different implementations:

- baseline (zjc)
- Baseline with hardware codec: Similar to baseline, but with both video decoding and encoding replaced with hardware-accelerated implementations (using NVENC for encoding and CUVID for decoding) for improved performance. (Jiache Zhang and Zijun Yang)
- CUDA: Similar to baseline with hardware codec, but with the rendering phase replaced with pixel-wise parallelism using CUDA kernels for high-performance computation of star trail effects. (Zijun Yang)

## PRELIMINARY RESULTS

> NOTICE: Most videos on this page are in 4K. If you notice any aliasing, viewing it in full screen usually helps.

### RENDERING ALGORITHMS

explain the source

some explanation of why we start from video instead of images (zjc)

- too large to store them in images

<video width="100%"   controls muted autoplay loop>
  <source src="https://github.com/zpatronus/StarTrailCUDA/raw/refs/heads/main/docs/videos/source.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video><center>Test video from Drew Cotten: <a href="https://www.youtube.com/watch?v=Bbp1-p2FoXU">https://www.youtube.com/watch?v=Bbp1-p2FoXU</a></center>

#### MAX Algorithm

<video width="100%"   controls muted autoplay loop>
  <source src="https://github.com/zpatronus/StarTrailCUDA/raw/refs/heads/main/docs/videos/max.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video><center>MAX Algorithm</center>
(zjc)

#### LINEAR Algorithm

<video width="100%"   controls muted autoplay loop>
  <source src="https://github.com/zpatronus/StarTrailCUDA/raw/refs/heads/main/docs/videos/linear.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video><center>LINEAR Algorithm</center>

(zjc)

#### LINEARAPPROX Algorithm

In the LINEAR algorithm, it is very expensive both in memory and computation to trace back $W$ input frames to calculate a single output frame, given that an ideal effect requires a window size of 30 to 120. To improve this, our first idea was to use exponential decay since it only needs the last output frame and the new input frame to generate the new output frame. However, it is very hard to control the tail length using exponential decay and the visual effect tends to be that the head is too bright while the tail is too dark.

To solve this issue, we leverage the fact that the exponential function has a derivative of 1 near $x=0$ and we can approximate a LINEAR decay with the following formula:

$$y=1+L-L e^{\frac{x}{WL}}$$

, where $L$ is some number larger than 5 and a greater $L$ gives a better approximation.

Here's a comparison between the ground truth (the blue line) and our heuristic (the green line) under $W=4$ and $L=10$.

![image-20251201110544491](./milestone.assets/image-20251201110544491.png)

With this heuristic, the new output frame can be approximated using only the last output frame and the new input frame using this formula in $O(1)$:

$$NewOutputFrame=\max\left((L+1)-(L+1-LastOutputFrame)e^{\frac{1}{LN}},NewInputFrame\right)$$

<video width="100%"   controls muted autoplay loop>
  <source src="https://github.com/zpatronus/StarTrailCUDA/raw/refs/heads/main/docs/videos/linearapprox.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video><center>LINEARAPPROX Algorithm</center>

The LINEARAPPROX algorithm looks very similar to the LINEAR algorithm in controlling the tail length and window size except for being slightly darker. We suspect the difference mainly comes from floating point precision and rounding errors. Also, both of us think it actually looks better than the LINEAR algorithm because it emphasizes bright stars and suppresses the darker ones, creating a cleaner video.

In the baseline implementation, we compared the performance difference for the LINEAR algorithm and the LINEARAPPROX algorithm under default parameters given the test input ($W=30$). The LINEAR algorithm takes 1125s to complete while the LINEARAPPROX only takes 152s, which is a huge 7.4x speedup gifted by math.

<video width="100%"   controls muted autoplay loop>
  <source src="https://github.com/zpatronus/StarTrailCUDA/raw/refs/heads/main/docs/videos/linearapprox_90w.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video><center>LINEARAPPROX Algorithm with a window size of 90 frames</center>

### BASELINE RENDERER

(zjc)

### BASELINE RENDERER WITH HARDWARE CODEC

Given that it takes a long time to encode output frames to output videos as seen in the baseline, and that we plan to use NVIDIA GPU, we leverage NVIDIA's codec library to decode and encode videos using hardware acceleration.

**Performance Results:**

Average time: 133.07s

**Timing Breakdown from a Typical Run:**

| Stage  | Time (ms) | Percentage |
| ------ | --------- | ---------- |
| Decode | 13,023.2  | 9.8%       |
| Render | 111,430   | 84.4%      |
| Encode | 7,673.19  | 5.8%       |
| Total  | 132,127   | 100%       |

The implementation shows significant improvement in encoding time compared to the baseline (reduced from 35.7s to 7.7s), demonstrating the efficiency of hardware-accelerated video processing.

The speedup so far is 8.5x

It is pretty clear that the bottleneck now is the rendering time.

### CUDA RENDERER

This implementation replaces the sequential CPU-based rendering computations with GPU-accelerated CUDA kernels that execute pixel-wise operations in parallel, dramatically reducing the rendering computation time.

**Performance Results:**

Average time: 40.94s

**Timing Breakdown from a Typical Run:**

| Stage  | Time (ms) | Percentage |
| ------ | --------- | ---------- |
| Decode | 19,553.5  | 39.1%      |
| Render | 12,100.1  | 24.2%      |
| Encode | 18,316.9  | 36.7%      |
| Total  | 49,970.4  | 100%       |

The CUDA implementation dramatically improves rendering time with a 9.2x speedup (from 111.4s to 12.1s). The total speedup compared to the original baseline reaches 27.4x.

Can we improve from here? Yes! We're working on it. Please refer to the PLAN FOR THE NEXT WEEK part.

## CURRENT CONCERNS

No major concerns. Just a matter of coding and doing the work. The main concern is our unfamiliarity with video codecs and their respective SDKs. There might be some pitfalls, but so far the progress looks good.

## PLAN FOR THE NEXT WEEK

explore pipelining and queuing, which is our 150% goal. 3 phase pipeline: decode-render-encode

aiming at around 10s given the current data

(zjc)

## EXPECTED DELIVERABLES AT THE END

(zjc)
