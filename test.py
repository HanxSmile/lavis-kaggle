import torchaudio as ta
from torchaudio.functional import resample as ta_resample
from df.logger import warn_once
from transformers import Wav2Vec2FeatureExtractor


def get_resample_params(method: str):
    params = {
        "sinc_fast": {"resampling_method": "sinc_interpolation", "lowpass_filter_width": 16},
        "sinc_best": {"resampling_method": "sinc_interpolation", "lowpass_filter_width": 64},
        "kaiser_fast": {
            "resampling_method": "kaiser_window",
            "lowpass_filter_width": 16,
            "rolloff": 0.85,
            "beta": 8.555504641634386,
        },
        "kaiser_best": {
            "resampling_method": "kaiser_window",
            "lowpass_filter_width": 16,
            "rolloff": 0.9475937167399596,
            "beta": 14.769656459379492,
        },
    }
    assert method in params.keys(), f"method must be one of {list(params.keys())}"
    return params[method]


def resample(audio, orig_sr: int, new_sr: int, method="sinc_fast"):
    params = get_resample_params(method)
    return ta_resample(audio, orig_sr, new_sr, **params)


def load_audio(
        file, sr=None, verbose=True, **kwargs
):
    ikwargs = {}
    if "format" in kwargs:
        ikwargs["format"] = kwargs["format"]
    rkwargs = {}

    audio, orig_sr = ta.load(file, **kwargs)
    if sr is not None and orig_sr != sr:
        if verbose:
            warn_once(
                f"Audio sampling rate does not match model sampling rate ({orig_sr}, {sr}). "
                "Resampling..."
            )
        audio = resample(audio, orig_sr, sr, **rkwargs)
    return audio.contiguous()
