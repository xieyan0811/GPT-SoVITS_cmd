import os
import shutil
from pydub import AudioSegment

ROOT_DIR = '/opt/xieyan/git/GPT-SoVITS_cmd/'

def set_speed(path_in, path_out, rate, format='mp3'):
    # demo: set_speed('/tmp/谢彦_new_fast.mp3', '/tmp/ooo.mp3', 0.8)
    if rate > 1.0:
        print('rate >1.0, speed up')
        if format == 'mp3':
            song = AudioSegment.from_mp3(path_in)
            fast_song = song.speedup(playback_speed=rate)
            fast_song.export(path_out, format="mp3")
        else:
            song = AudioSegment.from_wav(path_in)
            fast_song = song.speedup(playback_speed=rate)
            fast_song.export(path_out, format="wav")
    else:
        print('rate <=1.0, copy file')
        if path_in != path_out:
            shutil.copyfile(path_in, path_out)

def do_asr(path_in):
    path_out = path_in.replace('.mp3', '.txt')
    cmd = f'python {ROOT_DIR}/tools/asr/funasr_asr_file.py --input {path_in} --output {path_out}'
    os.system(cmd)
    with open(path_out, 'r') as f:
        text = f.read()
    os.remove(path_out)
    return text