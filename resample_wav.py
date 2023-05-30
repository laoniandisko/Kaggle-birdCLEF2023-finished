import os
import traceback
import warnings

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.simplefilter("ignore")

import librosa
import soundfile as sf
import os
import time

from multiprocessing import Pool

from tqdm import tqdm

def resample_wav(args):
    input_data_dir, output_data_dir, folder, file, target_sr = args
    audio_folder = os.path.join(output_data_dir, folder)
    out_path_ogg = os.path.join(audio_folder, f"{os.path.splitext(file)[0]}.ogg")
    out_path_wav = os.path.join(audio_folder, f"{os.path.splitext(file)[0]}.wav")
    exists_not_empty_ogg = os.path.exists(out_path_ogg) and os.path.getsize(out_path_ogg) > 20000
    exists_not_empty_wav = os.path.exists(out_path_wav) and os.path.getsize(out_path_wav) > 20000
    if not exists_not_empty_ogg and not exists_not_empty_wav:
        input_path = f'{input_data_dir}/{folder}/{file}'
        try:
            max_duration = 120
            duration_secs = librosa.get_duration(filename=input_path, sr=None)
            duration = max_duration if duration_secs > max_duration else None
            wav, sr = librosa.load(input_path, sr=None, duration=duration)
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            os.makedirs(audio_folder, exist_ok=True)
            try:
                os.remove(out_path_ogg)
            except:
                pass
            sf.write(
                file=out_path_wav,
                data=wav,
                format="wav",
                samplerate=target_sr
            )
        except Exception as e:
            print(f"Cannot process file {input_path}")
            traceback.print_stack()

if __name__ == '__main__':
    input_data_dir = '/mnt/md0/datasets/shared/train_audio_merged_adddata_v3_mauparfix'
    output_data_dir = '/mnt/md0/datasets/shared/train_audio_merged_adddata_v3_mauparfix_resampled'

    folders = os.listdir(input_data_dir)
    number_of_folders = len(folders)
    number_of_files = {}
    for folder in folders:
        number_of_files[folder] = len(os.listdir(f'{input_data_dir}/{folder}'))
        os.makedirs(f'{output_data_dir}/{folder}', exist_ok=True)

    target_sr = 32000
    search_space = []

    for index in range(number_of_folders):
        index_of_folder = index
        folder = folders[index_of_folder]

        for index_of_file in range(number_of_files[folder]):
            file = os.listdir(f'{input_data_dir}/{folder}')[index_of_file]

            search_space.append(
                (
                    input_data_dir,
                    output_data_dir,
                    folder,
                    file,
                    target_sr
                )
            )

    start_time = time.time()
    with Pool(processes=62) as pool:
        with tqdm(total=len(search_space), desc="Resampling") as pbar:
            for _ in enumerate(pool.imap_unordered(resample_wav, search_space)):
                pbar.update()
