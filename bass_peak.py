import librosa
import librosa.display
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

y, sr = librosa.load("./output/basscut1.wav", sr=44100, mono=True)

o_env = librosa.onset.onset_strength(y, sr=sr)**2
times = librosa.times_like(o_env, sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

plt.figure()
plt.subplot(2, 1, 1)
librosa.display.waveplot(y, sr=sr, alpha=0.25)
plt.title('Sectrogram')
plt.subplot(2, 1, 2)
plt.plot(times, o_env, label='Onset strength')
plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
           linestyle='--', label='Onsets')

plt.legend(frameon=True, framealpha=0.75)
plt.tight_layout()
plt.show()