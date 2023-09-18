import torch
import argparse
import librosa
import numpy as np
from panns_inference import AudioTagging, SoundEventDetection, labels

at = AudioTagging(checkpoint_path=None, device="cuda")


def get_features(args, feature_duration=2, sr=32000):
    audio, _sr = librosa.core.load(args.input, sr=sr, mono=True)
    time = audio.shape[-1] / sr
    batches = int(time // feature_duration)
    clip_sr = round(sr * feature_duration)
    assert clip_sr >= 9920
    audio_clips = np.reshape(audio[:batches * clip_sr], [batches, clip_sr])
    clipwise_output, embedding = at.inference(audio_clips[:10])
    features = embedding.data.cpu().numpy()
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", default='../data/1000053838/audio.mp4')
    parser.add_argument("-output", default='../data/1000053838/audio_panns.npz')
    args = parser.parse_args()

    feat = get_features(args)
    np.savez(feat, args.output)


if __name__ == "__main__":
    main()
