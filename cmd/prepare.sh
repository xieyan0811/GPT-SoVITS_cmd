#path=/opt/xieyan/tmp/audio/vocal/vocal_shenlei_3.mp3_10.mp3
#name=x1
path=$1
name=$2
echo "path" $path
echo "name" $name

/usr/local/bin/python tools/slice_audio.py $path output/slicer_opt/$name -34 4000 300 10 500 0.9 0.25 0 1
/usr/local/bin/python tools/cmd-denoise.py -i output/slicer_opt/$name -o output/denoise_opt/$name -p float32
/usr/local/bin/python tools/asr/funasr_asr.py -i output/denoise_opt/$name -o output/asr_opt/$name -s large -l zh -p float32

