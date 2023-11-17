# use soft link to unify audio filename
# because the original files has different suffixes, e.g. en.mp4, hin.mp4 ...

from pathlib import Path

root = Path("/home/ubuntu/short_form/data/")


def coalesce(content_id):
    audio_list = ["en", "hin", "hi"]
    for lang in audio_list:
        path = root / f"{content_id}/{content_id}_audio_{lang}.mp4"
        if path.is_file():
            return path


with open("select_audio.sh", "w") as f:
    for p in root.glob("*/*.json"):
        content_id = p.stem
        audio = coalesce(content_id)
        if not audio:
            continue
        cmd = f'ln -s {audio} {audio.parent / "audio.mp4"}\n'
        f.write(cmd)
