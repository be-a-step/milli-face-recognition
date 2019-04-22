import cv2
import os
import numpy as np
from PIL import Image
from pathlib import Path


def triming_images(source: str, dist: str, margin, cascade_conf_path: str):
    path = Path(source)
    face_cascade = get_cascade(Path(cascade_conf_path))
    for file_path in path.glob('**/*.png'):
        triming(file_path, str(
            file_path.parent).replace(str(path), dist), margin, face_cascade)


def triming(path: Path, dist: str, margin, face_cascade):
    if path.is_dir():
        return
    if not Path(dist).exists():
        Path(dist).mkdir(parents=True)

    # 画像を読み込む
    image_pil = Image.open(path.as_posix())
    # NumPyの配列に格納
    image = np.array(image_pil, 'uint8')
    # アニメ顔特徴分類器で顔を検知
    faces = face_cascade.detectMultiScale(image)
    # 検出した顔画像の処理
    for (x, y, w, h) in faces:
        # 顔画像領域のマージン設定
        resize_position = get_resize_position(
            x, y, w, h, margin, image_pil.width, image_pil.height)

        # 顔を 64x64 サイズにリサイズ
        roi = cv2.resize(image[resize_position[1]: resize_position[3], resize_position[0]: resize_position[2]], (64, 64), interpolation=cv2.INTER_LINEAR)
        # ファイル名を配列に格納
        filename = os.path.splitext(path.name)[0]
        save_path = Path(dist).joinpath(filename + ".png").as_posix()
        # そのまま保存すると青みがかる（RGBになっていない）
        cv2.imwrite(save_path, roi[:, :, ::-1].copy())


def get_cascade(path: Path):
    # アニメ顔特徴分類器
    face_cascade = cv2.CascadeClassifier(str(path))
    return face_cascade


def get_resize_position(x, y, w, h, margin, max_width, max_hight):
    margin_left = int(w * margin[0] / 100)
    margin_top = int(h * margin[1] / 100)
    margin_right = int(w * margin[2] / 100)
    margin_bottom = int(h * margin[3] / 100)

    pos_top = y - margin_top if y - margin_top > 0 else 0
    pos_bottom = y + h + margin_bottom if y + h + \
        margin_bottom < max_hight else max_hight
    pos_left = x - margin_left if x - margin_bottom > 0 else 0
    pos_right = x + w + margin_right if x + w + \
        margin_right < max_hight else max_width

    return (pos_left, pos_top, pos_right, pos_bottom)
