import cv2
from pathlib import Path
import os


def sampling_from_movies(source: str, dist: str, sampling_rate: int):
    path = Path(source)
    for filePath in path.glob('**/*'):
        sampling(filePath, str(
            filePath.parent).replace(str(path), dist), sampling_rate)


def sampling_from_movie(source: str, dist: str, sampling_rate: int):
    sampling(Path(source), dist, sampling_rate)


def sampling(path: Path, dist: str, sampling_rate: int):
    if path.is_dir():
        return
    if not Path(dist).exists():
        Path(dist).mkdir(parents=True)
    num = 0
    frame_count = 0
    prefix = os.path.splitext(path.name)[0]
    cap = cv2.VideoCapture(str(path))
    while cap.isOpened():
        if frame_count % sampling_rate is 0:
            ret, frame = cap.read()
            if ret is True:
                cv2.imwrite(Path(dist).joinpath(
                    prefix + "_{:0=5}".format(num) + ".png").as_posix(), frame)
                num += 1
            else:
                break
        frame_count += 1
    cap.release()
