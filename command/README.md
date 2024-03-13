# 代码说明
## 修改代码
get_best_prompt.py 加代码：用于选择合适的提示音频	
serv.py	加代码：启我的服务，把api.py拆开
inference.py	加代码：我的服务调用的推理相关的，从api.py中拆出来

## 脚本
prepare.sh	# 加：准备数据的脚本
train.sh	# 加：训练模型调用的脚本

# 训练模型
* 准备2min左右的音频数据 
* 拆背景音乐和前景人声 利用webui界面
* 准备：. xieyan/prepare.sh
* 调整识别不准的语音 利用webui界面
* 取最适合的音频提示 python xieyan/get_best_prompt.py
* 训练：. xieyan/train.sh
* 推理：python xieyan/serv.py -m 模型名

# 运行服务
uvicorn command.serv:app --host 0.0.0.0 --port 9880 --workers 3
