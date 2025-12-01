# **StarTrailCUDA: GPU-Accelerated Rendering of Night-Sky Star Trails Video** - Milestone Report

**Team Members:** Zijun Yang <zijuny@andrew.cmu.edu> and Jiache Zhang <jiachez@andrew.cmu.edu>

**Project Home Page:** [https://blog.zjyang.dev/StarTrailCUDA/](https://blog.zjyang.dev/StarTrailCUDA/)

![image-20251117192727072](./project_proposal.assets/image-20251117192727072.png)

<center><a href="https://www.bilibili.com/video/BV1Q64y1a7FE/?share_source=copy_web&vd_source=248bf19a901960bb7bbfb1705c664b9c&t=79">https://www.bilibili.com/video/BV1Q64y1a7FE/?share_source=copy_web&vd_source=248bf19a901960bb7bbfb1705c664b9c&t=79</a></center>

## WORK COMPLETED SO FAR

explore different algo:

- max (zjc)
- average (zjc)
- exponential (zjc)
  - hard to control tail length (zjc)
- linear (zjc)
- linear approx
- baseline (zjc)
- baseline with hardware codec
- cuda

## PRELIMINARY RESULTS 

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

<video width="100%"   controls muted autoplay loop>
  <source src="https://github.com/zpatronus/StarTrailCUDA/raw/refs/heads/main/docs/videos/linearapprox.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video><center>LINEARAPPROX Algorithm</center>

<video width="100%"   controls muted autoplay loop>
  <source src="https://github.com/zpatronus/StarTrailCUDA/raw/refs/heads/main/docs/videos/linearapprox_90w.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video><center>LINEARAPPROX Algorithm with a window size of 90 frames</center>
### BASELINE RENDERER

(zjc)

### BASELINE RENDERER WITH HARDWARE CODEC

### CUDA RENDERER

## CURRENT CONCERNS

## PLAN FOR THE NEXT WEEK

explore pipelining and queuing, which is our 150% goal. 3 phase pipeline: decode-render-encode

aiming at around 10s given the current data

(zjc)

## EXPECTED DELIVERABLES AT THE END

(zjc)
