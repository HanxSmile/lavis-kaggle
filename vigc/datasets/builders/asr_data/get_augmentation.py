from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    Compose,
    Gain,
    OneOf,
    PitchShift,
    PolarityInversion,
    TimeStretch,
)


def get_augmentation(musan_dir=None):
    transform = Compose(
        [
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2, leave_length_unchanged=False),
            Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.1),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
            OneOf(
                [
                    AddBackgroundNoise(sounds_path=musan_dir, min_snr_in_db=1.0, max_snr_in_db=5.0,
                                       noise_transform=PolarityInversion(), p=1.0),
                    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
                ] if musan_dir is not None else [
                    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0), ],
                p=0.2,
            ),
        ]
    )
    return transform
