import numpy as np
from scipy import signal
from scipy import fftpack
import soundfile as sf
from matplotlib import pyplot as plt

# オーバーラップをかける関数
def ov(data, samplerate, Fs, overlap):
    Ts = len(data) / samplerate                     # 全データ長
    Fc = Fs / samplerate                            # フレーム周期
    x_ol = Fs * (1 - (overlap / 100))               # オーバーラップ時のフレームずらし幅
    N_ave = int((Ts - (Fc * (overlap / 100))) /
                (Fc * (1 - (overlap / 100))))       # 抽出するフレーム数（平均化に使うデータ個数）

    array = []                                      # 抽出したデータを入れる空配列の定義

    # forループでデータを抽出
    for i in range(N_ave):
        ps = int(x_ol * i)                          # 切り出し位置をループ毎に更新
        array.append(data[ps:ps + Fs:1])            # 切り出し位置psからフレームサイズ分抽出して配列に追加
        final_time = (ps + Fs)/samplerate           # 切り出したデータの最終時刻
    return array, N_ave, final_time                 # オーバーラップ抽出されたデータ配列とデータ個数、最終時間を戻り値にする

# ハニング窓をかける関数（振幅補正係数計算付き）
def hanning(data_array, Fs, N_ave):
    han = signal.hann(Fs)                           # ハニング窓作成
    acf = 1 / (sum(han) / Fs)                       # 振幅補正係数(Amplitude Correction Factor)

    # オーバーラップされた複数時間波形全てに窓関数をかける
    for i in range(N_ave):
        data_array[i] = data_array[i] * han         # 窓関数をかける

    return data_array, acf

# dB(デシベル）演算する関数
def db(x, dBref):
    with np.errstate(divide='ignore'):
        y = 20 * np.log10(x / dBref)                   # 変換式
    return y                                       # dB値を返す

# 聴感補正(A補正)する関数
def aweightings(f):
    if f[0] == 0:
        f[0] = 1
    ra = (np.power(12194, 2) * np.power(f, 4))/\
         ((np.power(f, 2) + np.power(20.6, 2)) *
          np.sqrt((np.power(f, 2) + np.power(107.7, 2)) *
                  (np.power(f, 2) + np.power(737.9, 2))) *
          (np.power(f, 2) + np.power(12194, 2)))
    a = 20 * np.log10(ra) + 2.00
    return a

# 平均化FFTする関数
def fft_ave(data_array, samplerate, Fs, N_ave, acf):
    fft_array = []
    fft_axis = np.linspace(0, samplerate, Fs)      # 周波数軸を作成
    a_scale = aweightings(fft_axis)                # 聴感補正曲線を計算

    # FFTをして配列にdBで追加、窓関数補正値をかけ、(Fs/2)の正規化を実施。
    for i in range(N_ave):
        fft_array.append(db
                        (acf * np.abs(fftpack.fft(data_array[i]) / (Fs / 2))
                        , 2e-5))

    fft_array = np.array(fft_array) + a_scale      # 型をndarrayに変換しA特性をかける
    fft_mean = np.mean(fft_array, axis=0)          # 全てのFFT波形の平均を計算

    return fft_array, fft_mean, fft_axis

# 波形(x, y)からn個のピークを幅wで検出する関数(xは0から始まる仕様）
def findpeaks(dx, y, n, w):
    index_all = list(signal.argrelmax(y, order=w))                  # scipyのピーク検出
    index = []                                                      # ピーク指標の空リスト
    peaks = []                                                      # ピーク値の空リスト

    # n個分のピーク情報(指標、値）を格納
    for i in range(n):
        # n個のピークに満たない場合は途中でループを抜ける（エラー処理）
        if i >= len(index_all[0]):
            break
        index.append(index_all[0][i])
        peaks.append(y[index_all[0][i]])

    # 個数の足りない分を0で埋める（エラー処理）-->最小値が0以下になるようなデータの場合は検討要
    if len(index) != n:
        index = index + ([0] * (n - len(index)))
        peaks = peaks + ([0] * (n - len(peaks)))

    index = np.array(index) * dx                                  # xの分解能dxをかけて指標を物理軸に変換
    peaks = np.array(peaks)
    return index, peaks

# スペクトログラムからピークを検出する関数
    # fft_array=スペクトログラム（転置前）
    # dt, df=スペクトログラムの時間分解能, 周波数分解能
    # num_peaks=1つの周波数軸で検出するピーク数
    # w=ノイズ対策用の幅（order）
    # max_peaks=最終的にスペクトログラムから抽出するピーク数（振幅の大きい順）
def findpeaks_2d(fft_array, dt, df, num_peaks, w, max_peaks):

    # ピーク情報を初期化する
    time_index = np.zeros((len(fft_array), num_peaks))
    freq_index = np.zeros((len(fft_array), num_peaks))
    freq_peaks = np.zeros((len(fft_array), num_peaks))

    # 各周波数軸毎にピークを検出する
    for i in range(len(fft_array)):
        index, peaks = findpeaks(df, fft_array[i], n=num_peaks, w=w)    # ピーク検出
        freq_peaks[i] = peaks                                           # 検出したピーク値(振幅)を格納
        freq_index[i] = index                                           # 検出したピーク位置(周波数)を格納
        time_index[i] = np.full(num_peaks, i) * dt                      # 検出したピーク位置(時間)を格納

    # 平坦化する
    freq_peaks = freq_peaks.ravel()
    freq_index = freq_index.ravel()
    time_index = time_index.ravel()

    # ピークの大きい順（降順）にソートする
    freq_peaks_sort = np.sort(freq_peaks)[::-1]
    freq_index_sort = freq_index[np.argsort(freq_peaks)[::-1]]
    time_index_sort = time_index[np.argsort(freq_peaks)[::-1]]

    return freq_index_sort[:max_peaks], freq_peaks_sort[:max_peaks], time_index_sort[:max_peaks]

path = './output/basscut1.wav'                         #ファイルパスを指定
data, samplerate = sf.read(path)            #wavファイルを読み込む
x = np.arange(0, len(data)) / samplerate    #波形生成のための時間軸の作成

# Fsとoverlapでスペクトログラムの分解能を調整する。
Fs = 4096                                   # フレームサイズ
overlap = 90                                # オーバーラップ率

# オーバーラップ抽出された時間波形配列
time_array, N_ave, final_time = ov(data, samplerate, Fs, overlap)

# ハニング窓関数をかける
time_array, acf = hanning(time_array, Fs, N_ave)

# FFTをかける
fft_array, fft_mean, fft_axis = fft_ave(time_array, samplerate, Fs, N_ave, acf)

# ピーク検出する
spec_dt = final_time / N_ave
freq_index, freq_peaks, time_index = findpeaks_2d(fft_array, spec_dt, fft_axis[1], num_peaks=100, w=10, max_peaks=50)

print(freq_peaks)
print(freq_index)
print(time_index)

# スペクトログラムで縦軸周波数、横軸時間にするためにデータを転置
fft_array = fft_array.T

# ここからグラフ描画
# グラフをオブジェクト指向で作成する。
fig = plt.figure()
ax1 = fig.add_subplot(111)

# データをプロットする。
im = ax1.imshow(fft_array,
                vmin=0, vmax=80,
                extent=[0, final_time, 0, samplerate],
                aspect='auto',
                cmap='jet')

# カラーバーを設定する。
cbar = fig.colorbar(im)
cbar.set_label('SPL [dBA]')

# ピークを描画する。
ax1.scatter(time_index, freq_index, s=250, facecolor='None', linewidth=3, edgecolors='yellow')

for i in range(len(time_index)):
    plt.text(time_index[i],
             freq_index[i] + 150,
             str(round(freq_peaks[i], 2)),
             fontsize=10, color='black',
             backgroundcolor='white',
             horizontalalignment='center')

# 軸設定する。
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Frequency [Hz]')

# スケールの設定をする。
ax1.set_xticks(np.arange(0, 50, 1))
ax1.set_yticks(np.arange(0, 20000, 500))
ax1.set_xlim(0, 5)
ax1.set_ylim(0, 2000)

# グラフを表示する。
plt.show()
plt.close()