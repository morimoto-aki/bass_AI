import numpy as np
import matplotlib.pyplot as plt
import wave
import warnings

warnings.simplefilter('ignore', category=RuntimeWarning)  # RuntimeWarningを無視扱いに設定

def ReadWavFile(FileName = "./output/bass.wav"):
    """
    wavファイルを読み込み、配列として返す
    """
    try:
        wr = wave.open(FileName, "r")
    except FileNotFoundError: #ファイルが存在しなかった場合
        print("[Error 404] No such file or directory: " + FileName)
        return 0
    data = wr.readframes(wr.getnframes())
    wr.close()
    x = np.frombuffer(data, dtype="int16") / float(2**15)
    return x


sampling_rate = 44100 #サンプリングレート
data = np.array(ReadWavFile("./output/bass.wav")) #wavデータを配列で保存

NFFT = 1024 # フレームの大きさ
OVERLAP = NFFT / 2 # 窓をずらした時のフレームの重なり具合. half shiftが一般的らしい
frame_length = data.shape[0] #全フレーム数
split_number = len(np.arange((NFFT / 2), frame_length, (NFFT - OVERLAP))) #楽曲の分割数

window = np.hamming(NFFT)  #窓関数

spec = np.zeros([split_number, NFFT // 4]) #転置状態で定義初期化

pos = 0

for fft_index in range(split_number):
    frame = data[int(pos):int(pos+NFFT)]
    if len(frame) == NFFT:
        windowed = window * frame  #窓関数をかける
        fft_result = np.fft.rfft(windowed)
         # real()関数で虚部を削除して、さらに高周波をカット（複素共役による鏡像のため不要）
        fft_data2 = np.real(fft_result[:int(len(fft_result)/2)]) 
        fft_data2 = np.log(fft_data2** 2)  # グラフで見やすくするために対数をとります

        for i in range(len(spec[fft_index])):
            spec[fft_index][-i-1] = fft_data2[i]

        pos += (NFFT - OVERLAP) #窓をずらす

# プロット
plt.imshow(spec.T, extent=[0, frame_length, 0, sampling_rate/2], aspect="auto")
plt.xlabel("time[s]")
plt.ylabel("frequency[Hz]")
plt.colorbar()
plt.ylim(0,10000)
plt.show()

