import sampling_image
import triming_face

cascade_conf_path = "./resources/lbpcascade_animeface.xml"
source_directory = './resources/source/movies'
images_dist = './resources/source/images'
sampling_rate = 1000
margin = (20, 20, 20, 0)

sampling_image.sampling_from_movies(
    source_directory, images_dist, sampling_rate)

traning_dist = './resources/training/faces'

triming_face.triming_images(
    images_dist, traning_dist, margin, cascade_conf_path)
