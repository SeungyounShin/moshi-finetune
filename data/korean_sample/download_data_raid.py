import argparse
from datasets import load_dataset, Audio
from tqdm import tqdm
import soundfile as sf
import os
import json

def download_data(args):
    dataset = load_dataset(args.hf_dataset_name, split="train")

    # def replace_path(example):
    #     # 'audio' 컬럼에 'path'가 있다면 교체
    #     old_path = example["audio"]
    #     example["audio"] = old_path.replace("/mnt/work/robin", "/raid/channel/tts/ko/100")
    #     return example

    # dataset = dataset.map(replace_path, num_proc=32)
    # dataset.push_to_hub('channelcorp/aihub_moshi_100_aligned', num_shards=8, embed_external_files=False)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000, mono=False))

    # Ensure directories exist
    data_dir = args.data_dir
    audio_dir = os.path.join(data_dir, "audio")
    text_dir = os.path.join(data_dir, "text")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        audio_class = sample["audio"]
        audio_array = audio_class["array"]  # (2, T)
        sr = audio_class["sampling_rate"]  # ex. 16000
        aligned_timestamps = sample["aligned_timestamps"]

        # Save audio (.wav)
        audio_filepath = os.path.join(audio_dir, f"{args.dataset_signiture}_{i}.wav")
        sf.write(audio_filepath, audio_array.T, sr)

        # Process and save text (.json)
        formatted_timestamps = [
            {
                "speaker": "A" if word_info["role"] == "assistant" else "B",
                "word": word_info["word"],
                "start": word_info["absolute_start"],
                "end": word_info["absolute_end"],
            }
            for word_info in aligned_timestamps
        ]

        text_filepath = os.path.join(text_dir, f"{args.dataset_signiture}_{i}.json")
        with open(text_filepath, 'w', encoding='utf-8') as f:
            json.dump(formatted_timestamps, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_name", type=str, default="channelcorp/aihub_moshi_71592_aligned")
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.getcwd(), "/raid/channel/tts/k-moshi-ds"))
    parser.add_argument("--dataset_signiture", type=str, default="71592")
    args = parser.parse_args()

    download_data(args)
