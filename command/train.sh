#path=/opt/xieyan/tmp/audio/vocal/vocal_shenlei_3.mp3_10.mp3
#name=x1
path=$1
name=$2
echo "path" $path
echo "name" $name

/usr/local/bin/python GPT_SoVITS/prepare_datasets/1-get-text.py output/asr_opt/$name/$name.list output/asr_opt/$name $name 0 1 0 logs/$name GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large False
mv logs/$name/2-name2text-0.txt logs/$name/2-name2text.txt
/usr/local/bin/python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py output/asr_opt/$name/$name.list output/denoise_opt/$name $name 0 1 0 GPT_SoVITS/pretrained_models/chinese-hubert-base logs/$name False
/usr/local/bin/python GPT_SoVITS/prepare_datasets/3-get-semantic.py output/asr_opt/$name/$name.list $name 0 1 0 logs/$name GPT_SoVITS/pretrained_models/s2G488k.pth GPT_SoVITS/configs/s2.json False
mv logs/$name/6-name2semantic-0.tsv logs/$name/6-name2semantic.tsv

mkdir logs/$name/logs_s2
sed s/xxx/$name/g /opt/xieyan/git/GPT-SoVITS_240222/TEMP/tmp_s2.json > /opt/xieyan/git/GPT-SoVITS_240222/TEMP/tmp_s2_$name.json
/usr/local/bin/python GPT_SoVITS/s2_train.py --config "/opt/xieyan/git/GPT-SoVITS_240222/TEMP/tmp_s2_$name.json"

sed s/xxx/$name/g /opt/xieyan/git/GPT-SoVITS_240222/TEMP/tmp_s1.yaml > /opt/xieyan/git/GPT-SoVITS_240222/TEMP/tmp_s1_$name.yaml
/usr/local/bin/python GPT_SoVITS/s1_train.py --config_file "/opt/xieyan/git/GPT-SoVITS_240222/TEMP/tmp_s1_$name.yaml"


