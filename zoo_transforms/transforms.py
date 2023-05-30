import audiomentations as AA

## Default augmentations from last years second place
set_1 = AA.Compose(
    [
        AA.AddBackgroundNoise(
            sounds_path="input/ff1010bird_nocall/nocall", min_snr_in_db=0, max_snr_in_db=3, p=0.5
        ),
        AA.AddBackgroundNoise(
            sounds_path="input/train_soundscapes/nocall", min_snr_in_db=0, max_snr_in_db=3, p=0.25
        ),
        AA.AddBackgroundNoise(
            sounds_path="input/aicrowd2020_noise_30sec/noise_30sec", min_snr_in_db=0, max_snr_in_db=3, p=0.25
        ),
    ]
)

## Denton inspired augmentations
set_2 = AA.Compose(
    [
        AA.AddBackgroundNoise(
            sounds_path="input/ff1010bird_nocall/nocall", min_snr_in_db=0, max_snr_in_db=40, p=0.75
        ),
        AA.AddBackgroundNoise(
            sounds_path="input/train_soundscapes/nocall", min_snr_in_db=0, max_snr_in_db=40, p=0.50
        ),
        AA.AddBackgroundNoise(
            sounds_path="input/aicrowd2020_noise_30sec/noise_30sec", min_snr_in_db=0, max_snr_in_db=40, p=0.50
        ),
        AA.Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5), # DEFAULT
        AA.Shift(min_fraction=0.2, max_fraction=0.2, p=0.5),   # 15 + (15 * 0.2) = 18 input audio length 
        AA.LowPassFilter(min_cutoff_freq=100, max_cutoff_freq=10000, p=0.5), # possibly incorrect values
    ]
)