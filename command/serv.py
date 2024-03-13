"""
# api.py usage

` python api.py -dr "123.wav" -dt "一二三。" -dl "zh" `

## 执行参数:

`-s` - `SoVITS模型路径, 可在 config.py 中指定`
`-g` - `GPT模型路径, 可在 config.py 中指定`

调用请求缺少参考音频时使用
`-dr` - `默认参考音频路径`
`-dt` - `默认参考音频文本`
`-dl` - `默认参考音频语种, "中文","英文","日文","zh","en","ja"`

`-d` - `推理设备, "cuda","cpu","mps"`
`-a` - `绑定地址, 默认"127.0.0.1"`
`-p` - `绑定端口, 默认9880, 可在 config.py 中指定`
`-fp` - `覆盖 config.py 使用全精度`
`-hp` - `覆盖 config.py 使用半精度`

`-hb` - `cnhubert路径`
`-b` - `bert路径`

## 调用:

### 推理

endpoint: `/`

使用执行参数指定的参考音频:
GET:
    `http://127.0.0.1:9880?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh`
POST:
```json
{
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh"
}
```

手动指定当次推理所使用的参考音频:
GET:
    `http://127.0.0.1:9880?refer_wav_path=123.wav&prompt_text=一二三。&prompt_language=zh&text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_language=zh`
POST:
```json
{
    "refer_wav_path": "123.wav",
    "prompt_text": "一二三。",
    "prompt_language": "zh",
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh"
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400


### 更换默认参考音频

endpoint: `/change_refer`

key与推理端一样

GET:
    `http://127.0.0.1:9880/change_refer?refer_wav_path=123.wav&prompt_text=一二三。&prompt_language=zh`
POST:
```json
{
    "refer_wav_path": "123.wav",
    "prompt_text": "一二三。",
    "prompt_language": "zh"
}
```

RESP:
成功: json, http code 200
失败: json, 400


### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
    `http://127.0.0.1:9880/control?command=restart`
POST:
```json
{
    "command": "restart"
}
```

RESP: 无

"""


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

from command.inference import get_tts_wav, g_infer

g_config = global_config.Config()

# AVAILABLE_COMPUTE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="GPT-SoVITS api")

parser.add_argument("-s", "--sovits_path", type=str, default=g_config.sovits_path, help="SoVITS模型路径")
parser.add_argument("-g", "--gpt_path", type=str, default=g_config.gpt_path, help="GPT模型路径")
parser.add_argument("-m", "--model_name", type=str, default="caicai", help="模型名称") # xieyan add
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

#args = parser.parse_args()
args = argparse.Namespace(model_name='caicai',
                          sovits_path=g_config.sovits_path,
                          gpt_path=g_config.gpt_path,
                          default_refer_path='',
                          default_refer_text='',
                          default_refer_language='',
                          device='cuda',
                          bind_addr='0.0.0.0',
                          port=9880,
                          full_precision=False,
                          half_precision=False,
                          hubert_path=g_config.cnhubert_path,
                          bert_path=g_config.bert_path)

g_infer.load(g_config, args)

port = args.port
host = args.bind_addr

def handle_control(command):
    '''
    重启/退出服务
    '''
    if command == "restart":
        os.execl(g_config.python_exec, g_config.python_exec, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)

def handle_change(path, text, language):
    '''
    修改默认参考音频
    '''    
    if g_infer.model_info.set_default_refer(path, text, language):
        return JSONResponse({"code": 0, "message": "Success"}, status_code=200)
    else:
        return JSONResponse({"code": 400, "message": '缺少任意一项以下参数: "path", "text", "language"'}, status_code=400)

def handle_set_model(gpt_path, sovits_path):
    print(f"gptpath {gpt_path} vitspath {sovits_path}")
    g_infer.model_info.sovits_path = sovits_path
    g_infer.load_sovits_weights()
    g_infer.model_info.gpt_path = gpt_path
    g_infer.load_gpt_weights()


def handle(refer_wav_path, prompt_text, prompt_language, text, text_language, model_name):
    '''
    合成音频
    '''
    if model_name is not None:
        g_infer.set_model_info(model_name)

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

        
    wav = BytesIO()
    with torch.no_grad():
        gen = get_tts_wav(
            refer_wav_path, prompt_text, prompt_language, text, text_language
        )
        if gen is not None:
            sampling_rate, audio_data = next(gen)
            sf.write(wav, audio_data, sampling_rate, format="wav")
            wav.seek(0)

            torch.cuda.empty_cache()
            if args.device == "mps":
                print('executed torch.mps.empty_cache()')
                torch.mps.empty_cache()
    return StreamingResponse(wav, media_type="audio/wav")

app = FastAPI()

@app.post("/set_model_name")
async def set_model_name(request: Request):
    json_post_raw = await request.json()
    if g_infer.set_model_info(json_post_raw.get("model_name")):
        return JSONResponse({"code": 0, "message": "Success"}, status_code=200)
    else:
        return JSONResponse({"code": 400, "message": "未找到相关文件"}, status_code=400)

@app.get("/set_model_name")
async def set_model_name(model_name: str = None):
    if g_infer.set_model_info(model_name):
        return JSONResponse({"code": 0, "message": "Success"}, status_code=200)
    else:
        return JSONResponse({"code": 400, "message": "未找到相关文件"}, status_code=400)

@app.post("/get_status")
async def get_status(request: Request):
    model_name = g_infer.model_info.model_name
    model_list = g_infer.get_model_list()
    return JSONResponse({"code": 0, "model_name": model_name, "workers": 3,
                         "model_list":model_list}, status_code=200)

@app.get("/get_status")
async def get_status():
    model_name = g_infer.model_info.model_name
    model_list = g_infer.get_model_list()
    return JSONResponse({"code": 0, "model_name": model_name, "workers": 3,
                         "model_list":model_list}, status_code=200)

@app.post("/set_model")
async def set_model(request: Request):
    json_post_raw = await request.json()
    handle_set_model(json_post_raw.get("gpt_model_path"),json_post_raw.get("sovits_model_path"))
    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)

@app.post("/control")
async def control(request: Request):
    json_post_raw = await request.json()
    return handle_control(json_post_raw.get("command"))


@app.get("/control")
async def control(command: str = None):
    return handle_control(command)


@app.post("/change_refer")
async def change_refer(request: Request):
    json_post_raw = await request.json()
    return handle_change(
        json_post_raw.get("refer_wav_path"),
        json_post_raw.get("prompt_text"),
        json_post_raw.get("prompt_language")
    )


@app.get("/change_refer")
async def change_refer(
        refer_wav_path: str = None,
        prompt_text: str = None,
        prompt_language: str = None
):
    return handle_change(refer_wav_path, prompt_text, prompt_language)


@app.post("/")
async def tts_endpoint(request: Request):
    json_post_raw = await request.json()
    return handle(
        json_post_raw.get("refer_wav_path"),
        json_post_raw.get("prompt_text"),
        json_post_raw.get("prompt_language"),
        json_post_raw.get("text"),
        json_post_raw.get("text_language"),
        json_post_raw.get("model_name"),
    )


@app.get("/")
async def tts_endpoint(
        refer_wav_path: str = None,
        prompt_text: str = None,
        prompt_language: str = None,
        text: str = None,
        text_language: str = None,
        model_name: str = None
):
    return handle(refer_wav_path, prompt_text, prompt_language, text, text_language, model_name)

if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port, workers=3)
