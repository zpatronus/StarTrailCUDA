#!/bin/bash

rsync -avz aws618:/home/ubuntu/star/src/baseline/output.mp4 ./baseline.mp4
rsync -avz aws618:/home/ubuntu/star/src/baseline_nvenc/output.mp4 ./baseline_nvenc.mp4