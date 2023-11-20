import argparse
import librosa
import numpy as np
from glob import glob
from pathlib import Path
from panns_inference import AudioTagging

model = AudioTagging(checkpoint_path=None, device="cuda")


def get_features(input_file, feature_duration=2, sr=32000):
    audio, _sr = librosa.core.load(input_file, sr=sr, mono=True)
    time = audio.shape[-1] / sr
    batches = int(time // feature_duration)
    clip_sr = round(sr * feature_duration)
    assert clip_sr >= 9920
    audio_clips = np.reshape(audio[: batches * clip_sr], [batches, clip_sr])
    step = 500
    lst = []
    for i in range(0, batches, step):
        clipwise_output, embedding = model.inference(audio_clips[i : i + step])
        lst.append(embedding)
    features = np.concatenate(lst)
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        default="../data/1000053838/audio.mp4",
        help="input audio mp3/mp4, can be glob like `audio/*.mp4` to run batch inference.",
    )
    parser.add_argument(
        "output",
        type=Path,
        default="../data/1000053838/",
        help="output directory. content_id will be parsed from input directory, "
        "therefore the final path is $output/$content_id/panns_feature.npz",
    )
    args = parser.parse_args()

    input_list = sorted(glob(args.input))
    for f in input_list:
        content_id = Path(f).parent.name
        output = args.output / content_id / "panns_feature.npz"
        if output.exists():
            print(output, "exists.")
            continue
        try:
            feat = get_features(f)
            np.savez(output, feat)
        except:
            print(output, "error")
            continue


if __name__ == "__main__":
    main()
