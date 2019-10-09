import io
import os
import sys
from urllib.request import urlopen
from queue import Queue

import numpy as np
import scipy.signal as signal

import sounddevice as sd
import soundfile as sf
import fire


DEFAULT_SOURCE = 'http://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav'


def load_audio(path_or_url):
    if os.path.exists(path_or_url):
        return sf.read(path_or_url)
    else:
        return sf.read(io.BytesIO(urlopen(path_or_url).read()))


def chirp(sample_rate: int, length: int = None, fl: float=100, fh: float=1000):
    length = length or sample_rate
    t = np.arange(length) / sample_rate
    return signal.chirp(t, fl, t[-1], fh)


def shift_in(arr: np.ndarray, shift_in_arr: np.ndarray):
    end = arr.size - shift_in_arr.size
    arr[:end] = arr[shift_in_arr.size:]
    arr[end:] = shift_in_arr


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def convert_to_single_channel(arr: np.ndarray):
    if 1 == len(arr.shape):
        return arr
    if 2 == len(arr.shape):
        return arr[:, 0]
    raise ValueError('unexpected audio format')


class AutoPESQ:
    def __init__(self, sample_file: str = DEFAULT_SOURCE):
        # read audio file
        data, sample_rate = load_audio(sample_file)
        assert sample_rate in (8000, 16000,), 'unsupported sample rate, valid: 8000, 16000)'
        self._sample_rate: int = sample_rate
        # force to signal channel, as ITU PESQ implementation doesn't support multi channels
        self._sample_audio: np.ndarray = convert_to_single_channel(data)
        # generate signal wave form
        _chirp = chirp(self._sample_rate, self._sample_rate * 1)
        self._chirp = _chirp - np.mean(_chirp)
        self._chirp_std = np.std(self._chirp)

        self._guard = np.zeros(self._sample_rate * 1)
        self._sample_audio_with_guard = np.concatenate((self._guard, self._chirp, self._guard, self._sample_audio))

    def play(self, device=None):
        sd.play(self._sample_audio_with_guard,
                self._sample_rate,
                blocksize=1024,
                loop=True,
                device=device)
        sd.wait()

    def score(self, device=None):
        # queue for inter thread communication
        q = Queue(maxsize=4)

        def input_audio_callback(indata: np.ndarray, frames, time, status):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=sys.stderr)
            q.put(indata.copy())

        # DSP Start
        blocksize = 1024
        with sd.InputStream(samplerate=self._sample_rate,
                            callback=input_audio_callback,
                            blocksize=blocksize,
                            device=device):
            # window size
            xcorr_w = self._chirp.size
            record_w = self._guard.size + self._sample_audio.size

            # initialize buffer
            buffer_size = self._chirp.size + self._guard.size * 2 + blocksize * 2

            in_buf = np.zeros(buffer_size)
            xcorr_buf = np.zeros(buffer_size)  # normalized cross correlate of chirp and in_buf

            # buffer for audio recording
            record_buf = np.zeros(record_w)

            while True:  # DSP loop
                indata: np.ndarray = convert_to_single_channel(q.get())
                assert blocksize == indata.size, 'indata size is not equal to blocksize!'

                shift_in(in_buf, indata)

                def _correlate(a: np.ndarray):
                    std = np.std(a)
                    if std < self._chirp_std * 1e-4:
                        return 0
                    return np.mean((a - np.mean(a)) * self._chirp) / (std * self._chirp_std)

                _2d = rolling_window(in_buf[1 - xcorr_w - blocksize:], xcorr_w)
                xcorr = np.apply_along_axis(_correlate, 1, _2d)
                shift_in(xcorr_buf, xcorr)
        # DSP End


if __name__ == '__main__':
    fire.Fire(AutoPESQ)
