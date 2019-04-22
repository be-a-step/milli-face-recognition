import sampling_image
import triming_face

cascade_conf_path = "./resources/conf/cv2-cascade-conf/lbpcascade_animeface.xml"
source_directory = './resources/sources/movies'
images_dist = './resources/sources/images'
sampling_rate = 10
margin = (20, 20, 20, 0)

sampling_image.sampling_from_movies(
    source_directory, images_dist, sampling_rate)

traning_dist = './resources/dataset/faces'

triming_face.triming_images(
    images_dist, traning_dist, margin, cascade_conf_path)
