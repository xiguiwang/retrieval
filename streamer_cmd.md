# Some video and image processing command

## dump and transfer a docker image:
docker save -o llm-tgt.tar opea/llm-tgi:latest

docker load -i /home/user/myapp_v1.0.tar

## read MD file in linux console:
   pandoc -t plain file.md | less

## split and combine files (for transfor small files with slow and unstable network)

split -b 500M  largeFiles SmallFile.

generate SmallFile.aa   generate SmallFile.ab generate SmallFile.ac ...
cat SmallFile.* >  RestorelargeFiles


### uninstall python setup files:
python setup.py install --record files.txt
Once you want to uninstall you can use xargs to do the removal:
xargs rm -rf < files.txt


### 7z unzip file with password：
7z x  zip_file_with_password.zip

### gdb print pointer array
p *byteArray@16


## Gstreamer and FFmpe command
### gdb print pointer array
Gstreame pipeline

numactl --physcpubind=0 --localalloc gst-launch-1.0 filesrc location=./dataset/mall.avi ! decodebin ! fpsdisplaysink text-overlay=false silent=false video-sink=fakesink sync=false fps-update-interval=1000 --verbose

gst-launch-1.0 filesrc location=./dataset/mall.avi ! decodebin !  videoconvert n-threads=2 ! video/x-raw,format=RGB !  fpsdisplaysink text-overlay=false silent=false video-sink=fakesink sync=false fps-update-interval=1000 --verbose

gst-launch-1.0 -vf filesrc location=./dataset/intel_approved.mp4 ! qtdemux ! identity drop-buffer-flags=0x00002000 ! vaapih264dec ! fakesink sync=false

gst-launch-1.0 -vf filesrc location=./dataset/intel_approved.mp4 ! qtdemux ! identity drop-buffer-flags=0x00002000 ! vaapih264dec ! vaapijpegenc ! fakesink sync=true

gst-launch-1.0 -vf filesrc location=./dataset/intel_approved.mp4 ! qtdemux ! identity drop-buffer-flags=0x00002000 ! vaapih264dec ! vaapijpegenc ! multifilesink location=test_%05d.jpg sync=true


## FFmpeg

ffmpeg -discard nokey -i ./dataset/intel_approved.mp4 -q:v 2 -vf select="eq(pict_type\,PICT_TYPE_I)" -vsync 0 frame%03d.jpg

ffmpeg -discard nokey -i ./dataset/intel_approved.mp4 -q:v 2 -vf select="eq(pict_type\,PICT_TYPE_I)*(lt(abs(t-14),3)+lt(abs(t-107),3)+lt(abs(t-2113),3))" -vsync 0 frame%03d.jpg

ffmpeg -skip_frame nokey -i ./dataset/intel_approved.mp4 -vsync 0 -frame_pts true out%d.png

numactl --physcpubind=0 --localalloc ffmpeg -y  -discard nokey -i ./dataset/intel_approved.mp4 -q:v 2 -vf select="eq(pict_type\,PICT_TYPE_I)" -vsync 0  -update 1  ram_fs/tst.jpg

ffmpeg -benchmark -i ./dataset/intel_approved.mp4 -f null -

ffmpeg -y  -i ./dataset/intel_approved.mp4 -vsync 0  -f null -


### 1. clip parts of a video
ffmpeg -i input.mp4 -ss 00:05:10 -to 00:15:30 -c:v copy -c:a copy output2.mp4


### 2. concat videos
ffmpeg -i part1.mp4 -vcodec copy -vbsf h264_mp4toannexb -acodec copy part1.ts
ffmpeg -i part2.mp4 -vcodec copy -vbsf h264_mp4toannexb -acodec copy part2.ts
cat part1.ts part2.ts > parts.ts
ffmpeg -y -i parts.ts -acodec copy -ar 44100 -ab 96k -coder ac -vbsf h264_mp4toannexb parts.mp4


### 3. concat multiple video files
Extract video part of into TS file, combine multiple TS file together then convert TS into mp4 files.

ffmpeg -i part1.mp4 -vcodec copy -vbsf h264_mp4toannexb -acodec copy part1.ts
ffmpeg -i part2.mp4 -vcodec copy -vbsf h264_mp4toannexb -acodec copy part2.ts
cat part1.ts part2.ts > parts.ts
ffmpeg -y -i parts.ts -acodec copy -ar 44100 -ab 96k -coder ac -vbsf h264_mp4toannexb parts.mp4


### 4. convert video to GIF file
ffmpeg -i ./2024-12-11_16-33-21.mp4 output.gif
