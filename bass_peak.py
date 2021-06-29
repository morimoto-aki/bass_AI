import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy import signal

samplerate = 44100 #サンプリングレート
Fs = 4096 #フレームサイズ
overlap = 50 #オーバーラップ率

#オーディオファイル読み込み
def ReadWav(Filename):
    wf = wave.open(Filename,"r")
    buf= wf.readframes(wf.getnframes())
    
    data = np.frombuffer(buf, dtype="int16")
    
    return data

#オーバラップ処理
def OverLap(data, samplerate, Fs, overlap):
    Ts = len(data) / samplerate
    Fc = Fs / samplerate
    x_ol = Fs * (1 - (overlap/100))
    N_ave = int((Ts - (Fc * (overlap/100))) / (Fc * (1-(overlap/100))))
 
    array = []
 
    for i in range(N_ave):
        ps = int(x_ol * i)
        array.append(data[ps:ps+Fs:1])
    return array, N_ave #波形の配列データarray、平均化回数（＝分割数）N_ave

# 窓関数処理（ハニング窓）
def hanning(data_array, Fs, N_ave):
    han = signal.hann(Fs)# ハニング窓作成
    print(type(han))
    print(han)
    acf = 1 / (sum(han) / Fs)# 振幅補正係数(Amplitude Correction Factor)
    
    # オーバーラップされた複数時間波形全てに窓関数をかける
    for i in range(N_ave):
        data_array[i] = data_array[i] * han[i]        # 窓関数をかける

    return data_array, acf


data = ReadWav("./output/basscut1.wav")
data_ol, N_ave = OverLap(data, samplerate, Fs, overlap)
t = np.arange(0, Fs)/samplerate

# ハニング窓関数をかける
time_array, acf = hanning(data, Fs, N_ave)

# 軸のラベルを設定する。
plt.xlabel('Time [s]')
plt.ylabel('Signal [V]')

print(N_ave)
# データのラベルと線の太さ、凡例の設置を行う。
for j in range(N_ave):
    plt.plot(t, data_ol[j], label='Ch.1', lw=1)

plt.show()
