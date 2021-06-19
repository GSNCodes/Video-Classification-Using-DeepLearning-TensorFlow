import os
import cv2
import config
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from utils import *

if __name__ == "__main__":

    model = load_model('trained_model.h5')

    output_directory = 'Output_Videos'
    os.makedirs(output_directory, exist_ok = True)

    ## If you want to download a video from YouTube
    # video_url = '<Enter the url link here>'
    # video_title = download_from_youtube(video_url, output_directory)
    # input_video_file_path = f'{output_directory}/{video_title}.mp4'

    input_video_file_path = 'video_test/combined-test.mp4'
    video_title = 'Final'

    window_size = 1
    output_video_file_path = f'{output_directory}/{video_title}-Output-WSize-{window_size}.mp4'
    predict_on_live_video(model, input_video_file_path, output_video_file_path, window_size)


    window_size = 25
    output_video_file_path = f'{output_directory}/{video_title}-Output-WSize-{window_size}.mp4'
    predict_on_live_video(model, input_video_file_path, output_video_file_path, window_size)


    # To get a single prediction for the entire video
    input_single_video_file_path = 'video_test/basket-ball-test.avi'
    predictions_frames_count = 50
    make_average_predictions(model, input_single_video_file_path, predictions_frames_count)
