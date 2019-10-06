import io
import os
from urllib.request import urlopen

import sounddevice as sd
import soundfile as sf
import fire


DEFAULT_SOURCE = 'http://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav'


class Score:
    def __init__(self, source_file: str = DEFAULT_SOURCE):
        self._source_file = source_file

    def _read(self):
        if os.path.exists(self._source_file):
            return sf.read(self._source_file, dtype='float32')
        else:
            return sf.read(io.BytesIO(urlopen(self._source_file).read()), dtype='float32')

    def play(self):
        data, sample_rate = self._read()
        sd.play(data, sample_rate)
        sd.wait()

    def score(self):
        pass


if __name__ == '__main__':
    fire.Fire(Score)
