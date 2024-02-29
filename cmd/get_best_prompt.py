BASEDIR = '/opt/xieyan/git/GPT-SoVITS_240222'

import os
import librosa

os.chdir(BASEDIR)

def get_best(path, debug=False):
    with open(path) as fp:
        best_idx = 0
        best = 100
        lines = fp.readlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            arr = line.split('|')
            if len(arr[-1]) <= 10:
                continue
            if len(arr[-1]) == 0 or arr[-1][-1] not in '。，？！,.?!':
                continue
            if debug:
                print(arr[0], arr[-1])
            y, sr = librosa.load(arr[0], sr=16000)
            snd_length = len(y)/16000
            str_length = round(len(arr[-1])/4.75,2)
            scale = round(snd_length/str_length,3)
            diff = abs(scale-1)
            if diff < best:
                if debug:
                    print("@@@@", diff, best)
                best = diff
                best_idx = idx
            if debug:
                print(snd_length, str_length, scale)
                print("\n")
        with open(f'{path}.best', 'w') as fp:
            if debug:
                print("\n\n")
                print(best_idx, best, lines[best_idx])
            fp.write(lines[best_idx])

#get_best('output/asr_opt/caicai/caicai.list', True)

# 寻找所有 list 文件
def test():
    debug = True
    for root, dirs, files in os.walk('output/asr_opt'):
        for file in files:
            if file.endswith('.list'):
                print(os.path.join(root, file))
                get_best(os.path.join(root, file), debug)

test()