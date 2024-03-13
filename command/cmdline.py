import argparse
import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import signal
import torch
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from io import BytesIO
import config as global_config

from inference import get_tts_wav, g_infer

g_config = global_config.Config()

# AVAILABLE_COMPUTE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="GPT-SoVITS api")

parser.add_argument("-s", "--sovits_path", type=str, default=g_config.sovits_path, help="SoVITS模型路径")
parser.add_argument("-g", "--gpt_path", type=str, default=g_config.gpt_path, help="GPT模型路径")
parser.add_argument("-m", "--model_name", type=str, default="", help="模型名称") # xieyan add
parser.add_argument("-dr", "--default_refer_path", type=str, default="", help="默认参考音频路径")
parser.add_argument("-dt", "--default_refer_text", type=str, default="", help="默认参考音频文本")
parser.add_argument("-dl", "--default_refer_language", type=str, default="", help="默认参考音频语种")

parser.add_argument("-d", "--device", type=str, default=g_config.infer_device, help="cuda / cpu / mps")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0")
parser.add_argument("-p", "--port", type=int, default=g_config.api_port, help="default: 9880")
parser.add_argument("-fp", "--full_precision", action="store_true", default=False, help="覆盖config.is_half为False, 使用全精度")
parser.add_argument("-hp", "--half_precision", action="store_true", default=False, help="覆盖config.is_half为True, 使用半精度")
# bool值的用法为 `python ./api.py -fp ...`
# 此时 full_precision==True, half_precision==False

parser.add_argument("-hb", "--hubert_path", type=str, default=g_config.cnhubert_path, help="覆盖config.cnhubert_path")
parser.add_argument("-b", "--bert_path", type=str, default=g_config.bert_path, help="覆盖config.bert_path")

args = parser.parse_args()
g_infer.load(g_config, args)

def handle(refer_wav_path, prompt_text, prompt_language, text, text_language):
    '''
    合成音频
    '''
    if (
            refer_wav_path == "" or refer_wav_path is None
            or prompt_text == "" or prompt_text is None
            or prompt_language == "" or prompt_language is None
    ):
        refer_wav_path, prompt_text, prompt_language = (
            g_infer.model_info.path,
            g_infer.model_info.text,
            g_infer.model_info.language,
        )
        if not g_infer.model_info.is_ready():
            return JSONResponse({"code": 400, "message": "未指定参考音频且接口无预设"}, status_code=400)

    with torch.no_grad():
        gen = get_tts_wav(
            refer_wav_path, prompt_text, prompt_language, text, text_language
        )
        sampling_rate, audio_data = next(gen)

    wav = BytesIO()
    sf.write(wav, audio_data, sampling_rate, format="wav")
    wav.seek(0)

    torch.cuda.empty_cache()
    if args.device == "mps":
        print('executed torch.mps.empty_cache()')
        torch.mps.empty_cache()
    # 保存wav成文件
    with open('output.wav', 'wb') as f:
        f.write(wav.read())
    
    return StreamingResponse(wav, media_type="audio/wav")

for i in range(5):
    handle(None, None, None, '代码补全就是根据代码的上下文预测代码缺失的部分，比如在IDE中，对鼠标位置的代码自动完成，文档自动生成等。', 'zh')