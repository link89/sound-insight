import io
import os
import sys
from datetime import datetime
from urllib.request import urlopen
from queue import Queue

import numpy as np
import scipy.signal as signal
from scipy.signal.windows import blackmanharris

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


def normalize(arr: np.ndarray, min_std=1e-10):
    std = np.std(arr)
    if std < min_std:
        return None
    return (arr - np.mean(arr)) / std


def norm_xcorr(arr: np.ndarray, norm_template: np.ndarray, min_std=1e-10):
    def _norm_xcorr(a: np.ndarray):
        std = np.std(a)
        if std < min_std:
            return 0
        return np.mean((a - np.mean(a)) * norm_template) / std
    _2d = rolling_window(arr, norm_template.size)
    return np.apply_along_axis(_norm_xcorr, 1, _2d)


def force_single_channel(arr: np.ndarray):
    if 1 == len(arr.shape):
        return arr
    if 2 == len(arr.shape):
        return arr[:, 0]
    raise ValueError('too many dimensions (2 at most)')


def filename_friendly_time_string(t=None):
    if t is None:
        t = datetime.now()
    return t.strftime("%Y%m%d%H%M%S")


class SoundWriter:
    def __init__(self, fp: sf.SoundFile, defer: int, total: int):
        self._fp = fp
        self._total = total
        self._defer = defer

        self._offset = -self._defer

    def reset(self):
        self._fp.seek(0)
        self._offset = -self._defer

    def close(self):
        self._fp.close()

    @property
    def closed(self):
        return self._fp.closed

    def write(self, arr: np.ndarray):
        self._offset += arr.size
        print(self._offset)
        if self._offset <= 0:
            pass
        elif self._offset <= arr.size:
            self._fp.write(arr[-self._offset:])
        elif self._offset <= self._total:
            self._fp.write(arr)
        else:
            self._fp.write(arr[: self._total - (self._offset - arr.size)])
            self._fp.close()


class AutoPESQ:
    def __init__(self, sample_file: str=DEFAULT_SOURCE, output_dir='./output/'):
        # read audio file
        data, sample_rate = load_audio(sample_file)
        assert sample_rate in (8000, 16000,), 'unsupported sample rate, valid: 8000, 16000)'
        self._sample_rate: int = sample_rate

        # force to signal channel, as ITU PESQ implementation doesn't support multi channels
        self._sample_audio: np.ndarray = force_single_channel(data)

        # generate signal wave form
        _chirp_size = self._sample_rate
        self._chirp = chirp(self._sample_rate, _chirp_size) * blackmanharris(_chirp_size)
        self._norm_chirp = normalize(self._chirp)

        self._guard = np.zeros(self._sample_rate * 1)
        self._sample_audio_with_guard = np.concatenate((self._guard, self._chirp, self._guard, self._sample_audio))

        # others
        self._output_dir = output_dir
        self._file_no = 0

    def play(self, device=None):
        sd.play(self._sample_audio_with_guard,
                self._sample_rate,
                blocksize=1024,
                loop=True,
                device=device)
        sd.wait()

    def score(self, device=None, norm_peak_threshold=0.5):
        os.makedirs(self._output_dir, exist_ok=True)

        # queue for inter thread communication
        q = Queue(maxsize=4)

        def input_audio_callback(indata: np.ndarray, frames, time, status):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=sys.stderr)
            q.put(indata.copy())

        # DSP Start
        blocksize = 1024
        assert self._guard.size > blocksize, "fix me if blocksize is great than guard"
        with sd.InputStream(samplerate=self._sample_rate,
                            callback=input_audio_callback,
                            blocksize=blocksize,
                            device=device):
            # audio recording
            record_writer: SoundWriter = None

            # peak detection
            peak = None

            # initialize buffer
            buffer_size = self._chirp.size + blocksize
            in_buf = np.zeros(buffer_size)
            try:
                while True:  # DSP loop
                    indata: np.ndarray = force_single_channel(q.get())
                    assert blocksize == indata.size, 'indata size is not equal to blocksize!'
                    shift_in(in_buf, indata)

                    xcorr = norm_xcorr(in_buf[1 - self._chirp.size - blocksize:], self._norm_chirp)
                    max_pos = xcorr.argmax()

                    if xcorr[max_pos] > norm_peak_threshold:
                        print(xcorr[max_pos])
                        if record_writer is None or record_writer.closed:  # recording not start yet
                            peak = xcorr[max_pos]
                            record_writer = self._create_record_writer()
                        elif peak < xcorr[max_pos]:  # another peak is detected while recording not stop
                            print('reset record')
                            peak = xcorr[max_pos]
                            record_writer.reset()

                    if record_writer is not None and not record_writer.closed:
                        record_writer.write(indata)
            finally:
                if record_writer is not None:
                    record_writer.close()
        # DSP End

    def _create_record_writer(self,  filename=None):
        self._file_no += 1
        print('create new record')
        if filename is None:
            filename = os.path.join(self._output_dir, '{}-{}.wav'.format(self._file_no, filename_friendly_time_string()))
        fp = sf.SoundFile(filename, mode='x', samplerate=self._sample_rate, channels=1)
        return SoundWriter(fp, self._guard.size // 2, self._sample_audio.size + self._guard.size)


if __name__ == '__main__':
    fire.Fire(AutoPESQ)
