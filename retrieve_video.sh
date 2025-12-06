#!/bin/bash

rsync -avz aws618:/home/ubuntu/star/src/baseline/output.mp4 ./baseline.mp4

rsync -avz aws618:/home/ubuntu/star/src/baseline_nvenc/output.mp4 ./baseline_nvenc.mp4

rsync -avz aws618:/home/ubuntu/star/src/cuda/output.mp4 ./cuda.mp4

rsync -avz aws618:/home/ubuntu/star/src/cuda_pipeline/output.mp4 ./cuda_pipeline.mp4

rsync -avz aws618:/home/ubuntu/star/src/cuda_zero_copy_buffer_pool/output.mp4 ./cuda_zero_copy.mp4