import os
import torch
import numpy as np
import LangSegment
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
import librosa
from time import time as ttime

from transformers import AutoModelForMaskedLM, AutoTokenizer
from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from module.mel_processing import spectrogram_torch
from my_utils import load_audio

#XIEYAN_TEST = False # xieyan debug
XIEYAN_TEST = True

class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)

def is_full(*items):  # 任意一项为空返回False
    for item in items:
        if item is None or item == "":
            return False
    return True

def is_empty(*items):  # 任意一项不为空返回False
    for item in items:
        if item is not None and item != "":
            return False
    return True

def find_file(dirname, keywords):
    '''
    在 dirname 下寻找文件，文件名包含所有 keywords
    '''
    for root, dirs, files in os.walk(dirname):
        for file in files:
            found = True
            for keyword in keywords:
                if keyword not in file:
                    found = False
                    break
            if found:
                return os.path.join(root, file)
    return None

class ModelInfo:
    def __init__(self, args, config) -> None:
        self.sovits_path = args.sovits_path
        self.gpt_path = args.gpt_path

        if self.sovits_path == "":
            self.sovits_path = config.pretrained_sovits_path
            print(f"[WARN] 未指定SoVITS模型路径, fallback后当前值: {self.sovits_path}")
        if self.gpt_path == "":
            self.gpt_path = config.pretrained_gpt_path
            print(f"[WARN] 未指定GPT模型路径, fallback后当前值: {self.gpt_path}")
        
        self.path = args.default_refer_path
        self.text = args.default_refer_text
        self.language = args.default_refer_language

        model_name = args.model_name
        if model_name != "": # 优先级最高
            self.set_model_info(model_name)

        if self.path == "" or self.text == "" or self.language == "":
            self.path, self.text, self.language = "", "", ""
            print("[INFO] 未指定默认参考音频")
        else:
            print(f"[INFO] 默认参考音频路径: {self.path}")
            print(f"[INFO] 默认参考音频文本: {self.text}")
            print(f"[INFO] 默认参考音频语种: {self.language}")

    def is_ready(self) -> bool:
        return is_full(self.path, self.text, self.language)

    def set_default_refer(self, path, text, language):
        if is_empty(path, text, language):
            return False

        if path != "" or path is not None:
            self.path = path
        if text != "" or text is not None:
            self.text = text
        if language != "" or language is not None:
            self.language = language

        print(f"[INFO] 当前默认参考音频路径: {self.path}")
        print(f"[INFO] 当前默认参考音频文本: {self.text}")
        print(f"[INFO] 当前默认参考音频语种: {self.language}")
        print(f"[INFO] is_ready: {self.is_ready()}")
        return True

    def set_model_info(self, model_name):
        '''
        获取模型相关信息
        '''
        now_dir = os.getcwd()
        gpt_path = find_file(os.path.join(now_dir, 'GPT_weights/'), [f'{model_name}-e15'])
        sovits_path = find_file(os.path.join(now_dir, 'SoVITS_weights/'), [f'{model_name}_e8'])
        list_path = find_file('output/asr_opt/', [f'{model_name}.list.best'])
        print('path', gpt_path, sovits_path, list_path)
        if list_path is not None and gpt_path is not None and sovits_path is not None:
            with open(list_path) as fp:
                line = fp.readline().strip()
                arr = line.split('|')
                path = arr[0]
                text = arr[-1]
                language = 'zh'
                self.gpt_path = gpt_path
                self.sovits_path = sovits_path
                self.path = path
                self.text = text
                self.language = language
                print(f"set_model {model_name} successed")
                return True
        print('set_model_info failed')
        return False

class InferenceModel:
    def __init__(self):
        self.device = None
        self.is_half = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.ssl_model = None
        self.hps = None
        self.vp_model = None
        self.t2s_hz = None
        self.t2s_max_sec = None
        self.t2s_model = None
        self.t2s_config = None
        
    def load(self, config, args):
        '''
        把全局变量改成局部变量
        '''
        self.model_info = ModelInfo(args, config)

        self.device = args.device
        self.is_half = config.is_half
        if args.full_precision:
            self.is_half = False
        if args.half_precision:
            self.is_half = True
        if args.full_precision and args.half_precision:
            self.is_half = config.is_half

        print(f"[INFO] 半精: {self.is_half}")

        cnhubert_base_path = args.hubert_path
        bert_path = args.bert_path

        cnhubert.cnhubert_base_path = cnhubert_base_path
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        if self.is_half:
            self.bert_model = self.bert_model.half().to(self.device)
        else:
            self.bert_model = self.bert_model.to(self.device)

        self.ssl_model = cnhubert.get_model()
        if self.is_half:
            self.ssl_model = self.ssl_model.half().to(self.device)
        else:
            self.ssl_model = self.ssl_model.to(self.device)

        self.load_sovits_weights()
        self.load_gpt_weights()
        
    def set_model_info(self, model_name):
        if self.model_info.set_model_info(model_name):
            self.load_sovits_weights()
            self.load_gpt_weights()
            return True
        return False

    def load_sovits_weights(self):
        sovits_path = self.model_info.sovits_path
        dict_s2 = torch.load(sovits_path, map_location="cpu")
        self.hps = dict_s2["config"]
        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )
        if ("pretrained" not in sovits_path):
            del self.vq_model.enc_q
        if self.is_half == True:
            self.vq_model = self.vq_model.half().to(self.device)
        else:
            self.vq_model = self.vq_model.to(self.device)
        self.vq_model.eval()
        print(self.vq_model.load_state_dict(dict_s2["weight"], strict=False))

    def load_gpt_weights(self):
        gpt_path = self.model_info.gpt_path
        self.t2s_hz = 50
        dict_s1 = torch.load(gpt_path, map_location="cpu")
        self.t2s_config = dict_s1["config"]
        self.t2s_max_sec = self.t2s_config["data"]["max_sec"]
        self.t2s_model = Text2SemanticLightningModule(self.t2s_config, "****", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])
        if self.is_half == True:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(self.device)
        self.t2s_model.eval()
        total = sum([param.nelement() for param in self.t2s_model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

g_infer = InferenceModel()

def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                             hps.data.win_length, center=False)
    return spec

dict_language = {
    "中文": "zh",
    "英文": "en",
    "日文": "ja",
    "ZH": "zh",
    "EN": "en",
    "JA": "ja",
    "zh": "zh",
    "en": "en",
    "ja": "ja"
}

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = g_infer.bert_tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(g_infer.device)  #####输入是long不用管精度问题，精度随bert_model
        res = g_infer.bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    print('word2ph', word2ph, 'text', text)
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # if(is_half==True):phone_level_feature=phone_level_feature.half()
    return phone_level_feature.T

def clean_text_inf(text, language):
    '''
    copy from inference_webui.py
    '''
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text

def get_bert_inf(phones, word2ph, norm_text, language):
    '''
    copy from inference_webui.py
    '''    
    language=language.replace("all_","")
    if language == "zh" and len(word2ph) > 0:
        bert = get_bert_feature(norm_text, word2ph).to(g_infer.device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if g_infer.is_half == True else torch.float32,
        ).to(g_infer.device)

    return bert

def get_phones_and_bert(text,language):
    '''
    copy from inference_webui.py
    '''    
    dtype=torch.float16 if g_infer.is_half == True else torch.float32
    if language in {"en","all_zh","all_ja"}:
        language = language.replace("all_","")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(g_infer.device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if g_infer.is_half == True else torch.float32,
            ).to(g_infer.device)
    elif language in {"zh", "ja","auto"}:
        textlist=[]
        langlist=[]
        LangSegment.setfilters(["zh","ja","en"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        if len(bert_list) == 0:
            return [],torch.zeros(1024, 0, dtype=dtype).to(g_infer.device),""#返回空
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    return phones,bert.to(dtype),norm_text

def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language):
    '''
    copy from api.py
    '''
    t0 = ttime()
    prompt_text = prompt_text.strip("\n")
    prompt_language, text = prompt_language, text.strip("\n")
    zero_wav = np.zeros(int(g_infer.hps.data.sampling_rate * 0.3), dtype=np.float16 if g_infer.is_half == True else np.float32)
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if (g_infer.is_half == True):
            wav16k = wav16k.half().to(g_infer.device)
            zero_wav_torch = zero_wav_torch.half().to(g_infer.device)
        else:
            wav16k = wav16k.to(g_infer.device)
            zero_wav_torch = zero_wav_torch.to(g_infer.device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = g_infer.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
        codes = g_infer.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
    t1 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_language)
    phones1 = cleaned_text_to_sequence(phones1)
    text = text.replace('\r','') # xieyan debug, empty line break
    texts = text.split("\n")
    audio_opt = []

    for text in texts:
        text = text.strip() # xieyan debug
        if len(text) == 0:
            continue	

        if not XIEYAN_TEST:
            phones2, word2ph2, norm_text2 = clean_text(text, text_language)
            if len(word2ph2) == 0:
                continue
            phones2 = cleaned_text_to_sequence(phones2)

        if (prompt_language == "zh"):
            bert1 = get_bert_feature(norm_text1, word2ph1).to(g_infer.device)
        else:
            bert1 = torch.zeros((1024, len(phones1)), dtype=torch.float16 if g_infer.is_half == True else torch.float32).to(
                g_infer.device)
            
        if not XIEYAN_TEST:
            if (text_language == "zh"):
                bert2 = get_bert_feature(norm_text2, word2ph2).to(g_infer.device)
            else:
                bert2 = torch.zeros((1024, len(phones2))).to(bert1)
        else:
            phones2,bert2,norm_text2 = get_phones_and_bert(text,text_language)

        if len(phones2) == 0: # xieyan debug
            continue

        bert = torch.cat([bert1, bert2], 1)

        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(g_infer.device).unsqueeze(0)
        bert = bert.to(g_infer.device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(g_infer.device)
        prompt = prompt_semantic.unsqueeze(0).to(g_infer.device)
        t2 = ttime()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = g_infer.t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=g_infer.t2s_config['inference']['top_k'],
                early_stop_num=g_infer.t2s_hz * g_infer.t2s_max_sec)
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(g_infer.hps, ref_wav_path)  # .to(device)
        if (g_infer.is_half == True):
            refer = refer.half().to(g_infer.device)
        else:
            refer = refer.to(g_infer.device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = \
            g_infer.vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(g_infer.device).unsqueeze(0),
                            refer).detach().cpu().numpy()[
                0, 0]  ###试试重建不带上prompt部分
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    yield g_infer.hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
