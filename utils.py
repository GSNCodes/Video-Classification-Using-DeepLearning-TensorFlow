import os
import cv2
import math
import pafy
import config
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def check_dataset():

	plt.figure(figsize = (30, 30))


	all_classes_names = os.listdir('UCF50')

	# Generate a random sample of images
	random_range = random.sample(range(len(all_classes_names)), 20)

	# print(random_range)

	# Iterate through all the random samples
	for counter, random_index in enumerate(random_range, 1):

	    selected_class_Name = all_classes_names[random_index]

	    video_files_names_list = os.listdir(f'UCF50/{selected_class_Name}')

	    selected_video_file_name = random.choice(video_files_names_list)

	    video_reader = cv2.VideoCapture(f'UCF50/{selected_class_Name}/{selected_video_file_name}')

	    _, bgr_frame = video_reader.read()

	    video_reader.release()

	    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

	    cv2.putText(rgb_frame, selected_class_Name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
	    

	    plt.subplot(5, 4, counter)
	    plt.imshow(rgb_frame)
	    plt.axis('off')

	plt.show()


def frames_extraction(video_path):

    frames_list = []
    
    image_height, image_width = config.IMG_HEIGHT, config.IMG_WIDTH

    video_reader = cv2.VideoCapture(video_path)

    while True:

        ret, frame = video_reader.read() 

        if not ret:
            break

        resized_frame = cv2.resize(frame, (image_height, image_width))
        
        normalized_frame = resized_frame / 255
        
        frames_list.append(normalized_frame)
    
    video_reader.release()

    return frames_list


def create_dataset():
    max_images_per_class = config.MAX_IMGS_PER_CLASS
    dataset_directory = config.DATA_DIR
    classes_list = config.CLASSES_LIST

    temp_features = [] 
    features = []
    labels = []
    
    for class_index, class_name in enumerate(classes_list):
        print(f'Extracting Data of Class: {class_name}')
        
        files_list = os.listdir(os.path.join(dataset_directory, class_name))

        for file_name in files_list:

            video_file_path = os.path.join(dataset_directory, class_name, file_name)

            frames = frames_extraction(video_file_path)

            temp_features.extend(frames)
        
        features.extend(random.sample(temp_features, max_images_per_class))

        labels.extend([class_index] * max_images_per_class)

        temp_features.clear()

    features = np.asarray(features)
    labels = np.array(labels)  

    return features, labels

def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):

  metric_value_1 = model_training_history.history[metric_name_1]
  metric_value_2 = model_training_history.history[metric_name_2]


  epochs = range(len(metric_value_1))
  
  # Plotting the Graph
  plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
  plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)
  
  # Adding title to the plot
  plt.title(str(plot_name))

  # Adding legend to the plot
  plt.legend()

  plt.show()


def download_from_youtube(youtube_video_url, output_directory):
    video = pafy.new(youtube_video_url)

    video_best = video.getbest()

    output_file_path = f'{output_directory}/{video.title}.mp4'

    video_best.download(filepath = output_file_path, quiet = True)

    return video.title


def predict_on_live_video(model, video_file_path, output_file_path, window_size):

    predicted_labels_probabilities_deque = deque(maxlen = window_size)

    video_reader = cv2.VideoCapture(video_file_path)

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    image_height = config.IMG_HEIGHT
    image_width = config.IMG_WIDTH
    classes_list = config.CLASSES_LIST

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (original_video_width, original_video_height))

    while True: 

        ret, frame = video_reader.read() 

        if not ret:
            break

        resized_frame = cv2.resize(frame, (image_height, image_width))
        
        normalized_frame = resized_frame / 255

        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]

        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        if len(predicted_labels_probabilities_deque) == window_size:

            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            predicted_class_name = classes_list[predicted_label]
          
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        video_writer.write(frame)

     
    video_reader.release()
    video_writer.release()

def make_average_predictions(model, video_file_path, predictions_frames_count):
    
    image_height = config.IMG_HEIGHT
    image_width = config.IMG_WIDTH
    classes_list = config.CLASSES_LIST
    model_output_size = config.NUM_CLASSES

    predicted_labels_probabilities_np = np.zeros((predictions_frames_count, model_output_size), dtype = np.float)

    video_reader = cv2.VideoCapture(video_file_path)

    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    skip_frames_window = video_frames_count // predictions_frames_count

    for frame_counter in range(predictions_frames_count): 

        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        _ , frame = video_reader.read() 

        resized_frame = cv2.resize(frame, (image_height, image_width))
        
        normalized_frame = resized_frame / 255

        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]

        predicted_labels_probabilities_np[frame_counter] = predicted_labels_probabilities

    predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

    predicted_labels_probabilities_averaged_sorted_indexes = np.argsort(predicted_labels_probabilities_averaged)[::-1]

    for predicted_label in predicted_labels_probabilities_averaged_sorted_indexes:

        predicted_class_name = classes_list[predicted_label]

        predicted_probability = predicted_labels_probabilities_averaged[predicted_label]

        print(f"CLASS NAME: {predicted_class_name}   AVERAGED PROBABILITY: {(predicted_probability*100):.2}")
    
    video_reader.release()