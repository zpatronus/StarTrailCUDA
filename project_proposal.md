# StarTrailMPI: Parallel Rendering of Night-Sky Star Trails with MPI

## TITLE

**StarTrailMPI: Parallel Rendering of Night-Sky Star Trails with MPI**  
**Team Members:** Zijun Yang (zijuny@andrew.cmu.edu) and Jiache Zhang (jiachez@andrew.cmu.edu)

## URL

Project web page: [TODO: Insert project web page URL]

---

## SUMMARY

We will build a parallel rendering pipeline that converts a large sequence of fixed-camera night-sky photographs (e.g., one image per minute over an entire night) into a star-trail time-lapse video. Our implementation will use MPI to distribute frames and intermediate compositing across processes on GHC machines (and optionally AWS), focusing on parallel image compositing, load balancing, and communication-efficient reductions. 

The core serial algorithm is conceptually simple: for each new frame, we blend it with an accumulated image using either alpha blending or a max-compositing rule, and we output a sequence of accumulated images that gradually reveal longer star trails. The project challenge is to scale this process to thousands of high-resolution frames under realistic I/O constraints, while preserving visual quality and exploring different parallelization strategies.

---

## BACKGROUND

Long-exposure “star trail” photographs and time-lapse videos are a popular way to visualize the apparent motion of stars across the night sky. Traditionally, a single long exposure is used, which can easily overexpose bright regions and is vulnerable to noise, sensor artifacts, and transient occlusions (e.g., airplanes, clouds). A more robust and flexible approach is to capture a sequence of shorter-exposure images from a fixed camera and then digitally composite these frames to create star trails and time-lapse animations.

A simple serial algorithm for generating star trails from a sequence of frames works as follows:

1. Start from an initial accumulated image `A_0` equal to the first frame.
2. For each new frame `F_t` (t = 1..T-1):
   - Optionally apply a brightness and noise filter to `F_t`.
   - Compute `A_t = alpha * A_{t-1} + (1 - alpha) * F_t` (per-pixel blending), or use a max-compositing rule  
     `A_t(x, y) = max(A_{t-1}(x, y), F_t(x, y))` on selected color channels to emphasize star trails.
3. Use the sequence `{A_t}` to generate a time-lapse video that gradually reveals longer and longer star trails.

This workload is naturally data-parallel in two dimensions:

- **Across pixels**: Each pixel’s new value depends only on its own previous history and the current frame’s pixel, not on neighboring pixels.
- **Across frames**: Different subsets of frames can be partially composited in parallel and then combined via tree-structured reductions.

The challenge for this project is to take this conceptually simple algorithm and design an implementation that (1) scales across many cores and processes using MPI, (2) effectively hides or manages I/O and communication overheads, and (3) supports higher-level features such as brightness normalization, thresholding, and robust outlier removal without losing performance.

We previously implemented a parallel image renderer in a course assignment, where we rendered many circles in parallel into a single image. That assignment focused mostly on per-pixel parallelism and atomic updates inside a shared image. In contrast, this project emphasizes distributed compositing across many frames and MPI processes, as well as careful management of I/O and communication.

---

## THE CHALLENGE

Although the per-pixel computations are simple, achieving good speedup for large image sequences on a distributed-memory MPI system introduces several challenges:

### Workload Characteristics

- **Frame-based reduction dependency**: The naive algorithm is inherently sequential in time: `A_t` depends on `A_{t-1}`. We must restructure the algorithm to enable parallelism across frames (e.g., by having each MPI rank composite a subset of frames locally and then performing a reduction across ranks) without changing the final result.
- **High I/O volume**: A night-long capture can easily contain thousands of high-resolution RAW or JPEG images. Loading these frames can dominate runtime if not overlapped with computation. The I/O pattern is streaming and mostly sequential, but it is large relative to memory.
- **Mostly regular memory access**: For compositing, memory access is regular and cache-friendly (linear scans over image buffers). This suggests the computation may be memory-bandwidth-bound, so we need to be careful to not let communication costs overwhelm any benefit from parallelism.
- **Optional irregular processing**: If we include outlier rejection (e.g., removing airplanes, satellites, or clouds) or per-pixel thresholding based on temporal statistics, the workload may include conditional branches and diverging per-pixel workloads.

### System Constraints and Parallelization Challenges

- **Distributed-memory compositing**: With MPI, each process will hold a subset of frames and compute a partial composite. We must design a scalable reduction scheme (e.g., tree-based `MPI_Reduce`/`Allreduce`) to combine partial composites into a global result for each time step or for a set of checkpoints.
- **Communication vs. computation trade-off**: A naive scheme that communicates a full-resolution image after every frame would be communication-dominated and scale poorly. We need to explore strategies such as:
  - batching multiple frames before communication,
  - using a logarithmic-depth reduction tree for compositing,
  - minimizing data movement by keeping compositing local as long as possible.
- **Synchronization overhead**: If we render every intermediate `A_t` exactly, we may require frequent global barriers to ensure all processes have contributed their partial composites. We need to decide where barriers are necessary and where we can tolerate eventual consistency (e.g., only synchronize at selected time steps used in the exported video).
- **Load balancing**: If the number of frames is not evenly divisible by the number of MPI ranks, or if frames have varying sizes / processing complexity (e.g., different camera settings or file formats), simple static partitioning may result in some ranks finishing earlier and waiting at barriers.
- **I/O and storage constraints on GHC machines**: On GHC machines, each user has a very small local storage quota (on the order of ~2 GB). This means large image sequences and generated output frames will need to live primarily on remote storage (e.g., AFS or departmental network storage), where access latency and bandwidth may dominate runtime. We must design experiments and instrumentation that clearly separate local computation time from remote I/O time and understand how networked storage impacts our parallel scaling.

By addressing these challenges, we hope to deepen our understanding of distributed reductions, communication-efficient parallel algorithms for image processing, and the practical performance limits of MPI-based rendering workloads under realistic I/O conditions.

---

## RESOURCES

- **Compute resources**:
  - We plan to use **GHC machines** (departmental Linux servers/workstations) as our primary development and experimental platform for MPI.  
  - If time and budget permit, we may also run selected experiments on **Amazon Web Services (AWS)** to evaluate performance when large input datasets can be staged on local SSD-backed storage.
  - [TODO: Add any specific GHC machine types or AWS instance types once we decide.]

- **Starter code and previous assignments**:
  - Our own solutions from Assignment 2 (parallel image rendering with circles) and Assignment 3/4 (if relevant) will serve as references for basic image data structures, file I/O patterns, and performance measurement infrastructure.
  - We will write the core star-trail compositing pipeline from scratch in C/C++ with MPI, but will reuse patterns and helper utilities where appropriate.

- **Input data**:
  - Publicly available night-sky time-lapse sequences and fixed-camera astrophotography image sets from astronomy observatories or astrophotography communities.  
  - As a fallback, we can extract frames from existing star-trail or night-sky time-lapse videos on platforms like YouTube (for personal academic use only), and batch-convert them into still images.

- **Software stack**:
  - MPI (e.g., MPICH or Open MPI) installed on GHC machines (and on AWS instances if we use them).
  - Standard C/C++ toolchain and image processing libraries (e.g., `stb_image` or `libpng`) for reading and writing images.
  - Python or `ffmpeg` for converting sequences of still images into output videos for the final demo.

- **References**:
  - Online tutorials and documentation on digital star trail compositing and astrophotography post-processing.  
  - MPI documentation and class notes for collective communication and reduction patterns.

We do not anticipate needing any special hardware beyond what is already available on GHC machines and standard cloud instances, but we may reach out to the course staff if we discover we need access to higher-core-count machines for scaling experiments.

---

## GOALS AND DELIVERABLES

### Plan to Achieve (Baseline Goals)

1. **Correct serial implementation of star-trail compositing**
   - Implement a robust serial reference that:
     - Loads a sequence of input images from disk.
     - Applies a configurable blending rule (e.g., linear alpha blend or per-channel max) to generate accumulated images `A_t`.
     - Outputs both the final composite and an image sequence suitable for time-lapse video generation.
   - Verify correctness visually on small sequences and ensure deterministic results.

2. **MPI-based parallel compositing across frames**
   - Design and implement an MPI program where:
     - Each rank loads and processes a subset of frames.
     - Ranks compute local partial composites for their subset.
     - The program combines local partial composites via an MPI reduction (e.g., tree-structured) to obtain the same final result as the serial implementation.
   - Use static partitioning initially, and measure speedup versus a single-process baseline for various numbers of ranks.

3. **Performance evaluation on realistic image sets**
   - Run experiments on several datasets with varying numbers of frames and resolutions (e.g., 1080p and higher, hundreds to thousands of frames).
   - Collect and plot:
     - Execution time vs. number of MPI ranks.
     - Speedup and efficiency curves.
     - Breakdown of time spent in I/O, computation, and communication.
   - Provide a qualitative and quantitative analysis of where the performance is limited (I/O, memory bandwidth, or communication), especially under the small-local-storage / remote-I/O constraints of GHC machines.

4. **Poster-session demo**
   - Prepare at least one visually appealing star-trail time-lapse video generated by our MPI implementation.
   - Prepare speedup and time-breakdown plots for display at the poster session.

### Hope to Achieve (Stretch Goals)

1. **Communication-efficient incremental compositing**
   - Explore strategies where ranks produce partial composites for contiguous time segments and only synchronize at selected time steps (e.g., every k frames) used in the final video.
   - Evaluate the trade-off between communication overhead and temporal resolution of the output.

2. **Improved visual quality features**
   - Implement optional per-pixel processing such as:
     - brightness/contrast normalization across frames,
     - threshold-based star enhancement (emphasizing bright stars while suppressing noise),
     - basic outlier rejection to remove transient artifacts (e.g., airplanes).
   - Measure how much additional computation these features introduce and how they affect parallel scalability.

3. **Hybrid data parallelism within each rank**
   - Inside each MPI rank, use OpenMP or vectorization to parallelize per-pixel operations over the local image tiles.
   - Compare “MPI only” vs. “MPI + OpenMP” performance.

### Goals if Things Go Slower Than Expected

If we encounter unexpected difficulties (e.g., with I/O, data formats, or MPI environment issues), we will scale down our goals as follows:

1. Restrict input resolutions and dataset sizes to smaller images and fewer frames to ensure we can still conduct meaningful experiments.
2. Focus on a single blending rule (e.g., max-compositing) rather than supporting multiple modes and advanced image processing features.
3. Limit experiments to single-node MPI runs (multiple processes on one machine) if multi-node experiments or AWS setup prove unreliable, while still providing a careful analysis of intra-node scaling and communication costs.

Even in the reduced-scope scenario, we will prioritize having:
- a working serial and MPI implementation, and
- a solid performance evaluation with clear analysis of bottlenecks.

---

## PLATFORM CHOICE

We plan to implement our renderer in C/C++ with MPI and run it primarily on GHC machines (departmental Linux servers/workstations) and, if needed, on Amazon Web Services (AWS) for larger-scale experiments.

- **GHC machines (primary platform)**:
  - **Advantages**:
    - Easily accessible with our existing course accounts and toolchains.
    - No monetary cost to run experiments.
    - Well-integrated with our development workflow and course environment (debugging, profiling, and job management).
  - **Disadvantages**:
    - Very limited local disk capacity (on the order of ~2 GB per user), which makes it difficult to store large collections of high-resolution input images and generated output videos locally.
    - We will likely need to store most input frames and intermediate outputs on remote storage (e.g., AFS, departmental network storage, or cloud storage). This introduces **additional I/O latency** over the network, which may become a significant bottleneck for our workload.
    - Because our per-pixel computation is relatively simple, the observed end-to-end performance may be dominated by network I/O and file system behavior rather than pure CPU throughput.

  To mitigate these issues, we plan to:
  - Carefully stage data (e.g., copy smaller subsets of frames to local scratch space when possible).
  - Measure and report I/O time separately from computation and communication.
  - Design experiments that highlight the parallel scaling of the compositing kernel itself, while still discussing realistic end-to-end performance under networked storage.

- **AWS (secondary / optional platform)**:
  - **Advantages**:
    - Access to instances with more cores, more memory, and larger local disks (e.g., SSD-backed ephemeral storage), which can reduce I/O bottlenecks when all image data is staged locally on the instance.
    - Flexibility to scale up or down depending on the size of the input dataset and the number of MPI ranks we wish to test.
  - **Disadvantages**:
    - Monetary cost and setup overhead (AMI configuration, security groups, MPI setup).
    - Less tightly integrated with our existing course environment and automation.

Our plan is to implement, debug, and validate the serial and MPI versions on GHC machines (where development is convenient and free). If time and budget permit, we may replicate key experiments on one or more AWS instances to compare performance in an environment with larger local storage and potentially higher aggregate I/O bandwidth. This comparison will help us understand how much of our performance is limited by networked storage versus CPU and MPI communication.

If time permits, we may optionally add OpenMP parallelism within each MPI rank to explore hybrid programming models and better utilize multi-core CPUs.

---

## SCHEDULE

We expect to have approximately four weeks to complete the project from proposal approval to the final deadline. Below is our compressed week-by-week schedule.

### Week 1 (Nov 17–Nov 23)

- Finalize data sources and decide on the image format (e.g., PNG or JPEG).
- Implement and validate the serial reference pipeline:
  - image loading and saving,
  - basic blending rules (alpha blend and per-channel max),
  - generation of a sequence of accumulated images suitable for time-lapse.
- Begin designing the MPI decomposition (frame partitioning, reduction strategy, communication frequency).
- Set up and test basic MPI environment on GHC machines.

### Week 2 (Nov 24–Nov 30)

- Implement the initial MPI version with static partitioning:
  - Each rank loads and processes a subset of frames.
  - Ranks compute local partial composites and participate in a reduction to produce the final composite image.
- Extend the MPI implementation to support generating selected intermediate accumulated images (for the video) if time permits.
- Add timing instrumentation to separate I/O, computation, and communication.
- Run small- to medium-scale experiments to verify correctness and obtain preliminary performance results on GHC machines.

### Week 3 (Dec 1–Dec 7)

- Optimize communication patterns (e.g., tree-based reductions, batched frame compositing) to reduce synchronization and data movement overhead.
- Explore simple load-balancing strategies if needed (e.g., adjusting frame assignments or basic dynamic scheduling).
- Implement at least one visual quality feature (e.g., brightness normalization or threshold-based star enhancement) in both serial and MPI versions.
- Collect more systematic performance data on larger datasets and multiple process counts.
- If feasible, perform initial experiments on an AWS instance to compare local-SSD-based performance with GHC’s remote-storage-based performance.

### Week 4 (Dec 8–Dec 14)

- Finalize the set of visual quality features and ensure the MPI implementation remains correct and stable.
- Run comprehensive performance experiments:
  - multiple dataset sizes and resolutions,
  - different numbers of MPI ranks (and nodes/instances if possible),
  - different compositing strategies (e.g., per-frame vs. batched synchronization).
- Generate final star-trail videos and choose the best examples for the demo.
- Prepare speedup, efficiency, and time-breakdown plots (including I/O vs computation vs communication).
- Prepare the poster (overview, algorithm, parallel design, experimental results, visuals).
- Write and finalize the final project report, incorporating insights from all experiments.