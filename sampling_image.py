import cv2
from pathlib import Path
import os
from io import TextIOWrapper

cache_file = "sampling_cache"


def sampling_from_movies(source: str, dist: str, sampling_rate: int):
    path = Path(source)
    if not os.path.isfile(cache_file):
        with open(cache_file, "w") as f:
            pass
    with open(cache_file, "r") as f:
        cache = f.readlines()
    with open(cache_file, "a") as f:
        for file_path in path.glob('**' + os.sep + '*'):
            if str(file_path) + "\n" in cache or ".gitkeep" in str(file_path):
                continue
            print("processing: " + str(file_path))
            sampling(
                file_path, str(
                    file_path.parent).replace(
                    str(path), dist), sampling_rate, f)


def sampling_from_movie(
        source: str,
        dist: str,
        sampling_rate: int,
        f: TextIOWrapper):
    sampling(Path(source), dist, sampling_rate, f)


def sampling(path: Path, dist: str, sampling_rate: int, f: TextIOWrapper):
    if path.is_dir():
        return
    if not Path(dist).exists():
        Path(dist).mkdir(parents=True)
    num = 0
    frame_count = 0
    prefix = os.path.splitext(path.name)[0]
    cap = cv2.VideoCapture(str(path))
    while cap.isOpened():
        ret, frame = cap.read()
        if frame_count % sampling_rate == 0:
            if ret is True:
                cv2.imwrite(Path(dist).joinpath(
                    prefix + "-{:0=5}".format(num) + ".png").as_posix(), frame)
                num += 1
            else:
                break
        frame_count += 1
    cap.release()
    f.write(str(path) + "\n")
