import io
import os
from urllib.request import urlopen

import numpy as np
import scipy.signal as signal
import sounddevice as sd
import soundfile as sf
import fire


DEFAULT_SOURCE = 'http://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav'


class AutoScore:
    def __init__(self, sample_file: str = DEFAULT_SOURCE):
        self._sample_file = sample_file

    def _read(self):
        if os.path.exists(self._sample_file):
            return sf.read(self._sample_file, dtype='float32')
        else:
            return sf.read(io.BytesIO(urlopen(self._sample_file).read()), dtype='float32')

    def _insert_chirp(self, data: np.ndarray, sample_rate: int, duration: float=1, guard: float=1, fl: float=100, fh: float=1000):
        t = np.arange(int(sample_rate * duration)) / sample_rate
        c = signal.chirp(t, fl, duration, fh)
        g = np.zeros(int(sample_rate * guard))
        o = np.concatenate((g, c, g, data))
        if len(data.shape) > 1:  # handle multi channels
            o = np.column_stack((o,) * data.shape[1])
        return o

    def play(self):
        data, sample_rate = self._read()
        data = self._insert_chirp(data, sample_rate)
        sd.play(data, sample_rate, blocksize=1024, loop=True)
        sd.wait()

    def score(self):
        pass


if __name__ == '__main__':
    fire.Fire(AutoScore)
