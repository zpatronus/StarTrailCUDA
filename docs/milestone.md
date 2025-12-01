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

#### Source Material Selection

Our source material is a high-quality all-night sky recording (non-time-lapse) from YouTube (video source: https://www.youtube.com/watch?v=Bbp1-p2FoXU). This video provides excellent clarity and resolution (3840×2160), making it ideal for rendering star trail videos.

We considered alternative sources, particularly examining the database at https://data.phys.ucalgary.ca/, which contains comprehensive continuous nighttime recordings from observatories worldwide over recent years. While this database offers extensive and complete data, it presents a significant limitation: the cameras are primarily scientific instruments designed for research purposes rather than high-quality cinematography, resulting in insufficient clarity for star trail rendering. Therefore, we opted for clear videos recorded by amateur astronomy enthusiasts.

#### Video vs. Image Format Consideration

Theoretically, using video files versus collections of individual images as source media would yield equivalent results. However, storing media in image format requires prohibitively large storage space. Our test video, when stored as an MP4 file, occupies 101.77MB. When each frame of this video is saved in PNG format, the total storage requirement reaches 18GB; even with JPEG compression, it still requires 3.6GB. This storage demand is unacceptable for development and testing environments with limited storage capacity (for example, GHC servers impose a 2GB storage limit per user, insufficient for image-format media storage). Mounting remote storage would introduce additional I/O overhead.

Consequently, we decided to store and process source media in video format to minimize local storage requirements. Additionally, MP4-compressed video maintains sufficient quality without affecting final rendering results, eliminating the need for uncompressed video or image inputs. 

<video width="100%"   controls muted autoplay loop>
  <source src="https://github.com/zpatronus/StarTrailCUDA/raw/refs/heads/main/docs/videos/source.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video><center>Test video from Drew Cotten: <a href="https://www.youtube.com/watch?v=Bbp1-p2FoXU">https://www.youtube.com/watch?v=Bbp1-p2FoXU</a></center>




#### MAX Algorithm

<video width="100%"   controls muted autoplay loop>
  <source src="https://github.com/zpatronus/StarTrailCUDA/raw/refs/heads/main/docs/videos/max.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video><center>MAX Algorithm</center>

The MAX algorithm creates star trails by maintaining a cumulative maximum frame that preserves the brightest pixel values encountered over time. At each frame, the algorithm performs an element-wise maximum operation between the current input frame and the accumulated maximum frame, ensuring that bright stars leave persistent trails while darker background regions remain unaffected.

Key implementation details:
- **Decay Factor**: A decay factor of 0.999 is applied to the accumulated frame before comparison, causing older trails to gradually fade and preventing infinite accumulation of brightness values
- **Pixel-wise Maximum**: For each pixel position, the algorithm selects the maximum value between the decayed accumulated frame and the current frame: `maxFrame = max(maxFrame × 0.999, currentFrame)`

While this approach is conceptually simple and straightforward to implement, the practical results are suboptimal. The decay factor of 0.999 is insufficient to prevent excessive accumulation of brightness values over time. As the video progresses, bright points fade too slowly, causing an increasing number of luminous artifacts to persist in the frame. This leads to severe overexposure and blurring effects, where the accumulated trails become overly bright and lose definition. The resulting star trail video appears washed out with poor contrast and indistinct trail boundaries, making it difficult to discern individual star trajectories. Although the MAX algorithm successfully demonstrates the basic concept of trail preservation, its aggressive retention of bright pixels without adequate decay control produces unsatisfactory visual quality.

#### LINEAR Algorithm

<video width="100%"   controls muted autoplay loop>
  <source src="https://github.com/zpatronus/StarTrailCUDA/raw/refs/heads/main/docs/videos/linear.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video><center>LINEAR Algorithm</center>

The LINEAR algorithm produces the highest quality star trail effects among all tested approaches by implementing a sophisticated sliding window technique with linear weight decay. This method maintains a fixed-size window of recent frames and applies linearly decreasing weights to create natural-looking trail fade effects.

Key implementation details:
- **Sliding Window**: Maintains a deque of the most recent W frames (where W is the window size), automatically removing the oldest frame when the window is full
- **Linear Weight Decay**: Applies weights ranging from 1.0 (most recent frame) to 1/W (oldest frame in window), creating a smooth linear fade: `weight = (W - i) / W`
- **Pixel-wise Maximum**: For each frame position, performs element-wise maximum operation between the weighted frame and the accumulated result: `accFrame = max(accFrame, weightedFrame)`

This approach produces visually superior results with well-defined star trajectories, appropriate trail lengths, and natural fade characteristics. The linear weighting ensures that recent star positions are prominent while older positions gradually diminish, creating realistic motion blur effects that closely resemble long-exposure photography techniques.

However, the algorithm's computational complexity is significantly higher due to the need to maintain and process the entire sliding window for each output frame. In our serial implementation, the LINEAR algorithm requires approximately 968 seconds to complete rendering, making it impractical for real-time or near-real-time applications. The performance bottleneck stems from the intensive memory operations and repeated maximum computations across the sliding window, highlighting the critical need for parallel optimization techniques.

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
