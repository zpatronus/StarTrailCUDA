#!/bin/bash

rsync -avz --exclude-from=rsync-exclude * aws618:~/star
rsync -avz src/video_download/test_starrail_15sec.mp4 src/video_download/test_starrail_15sec_h264.mp4 aws618:~/star/src/video_download