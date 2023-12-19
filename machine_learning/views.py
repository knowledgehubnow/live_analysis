from django.shortcuts import render, redirect,HttpResponse
import cv2
import json
from django.core.serializers import serialize
import numpy as np
import speech_recognition as sr
import subprocess
from datetime import datetime
from deepface import DeepFace
from .models import *
import ast
import time
from pdfminer.high_level import extract_text
from io import BytesIO
from pydub import AudioSegment  # Import AudioSegment for voice modulation analysis
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer  # Import SentimentIntensityAnalyzer for sentiment analysis
import mediapipe as mp
from math import degrees, acos
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import librosa
from tensorflow import keras
import dlib
import tensorflow as tf
import imutils 
from scipy.spatial import distance as dist 
from imutils import face_utils 
import pickle
import sys
import os
from voice_emotion import extract_feature
import re
from tensorflow.keras.models import load_model
from pydub.silence import split_on_silence
import math as m
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import *
from rest_framework.parsers import FormParser, MultiPartParser
import uuid
from .body_posture_detection import body_posture as detect_body_posture
from moviepy.editor import VideoFileClip, AudioFileClip
import pyaudio
import threading
import wave


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


detector = dlib.get_frontal_face_detector() 
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(project_root_dir, 'shape_predictor_68_face_landmarks.dat')
landmark_predict = dlib.shape_predictor(folder_path)

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()

def preprocess_frame(frame):
    # Resize to match the input shape of your model
    resized_frame = cv2.resize(frame, (48, 48))
    # Convert to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)   
    # Normalize pixel values to be between 0 and 1
    normalized_frame = gray_frame / 255.0   
    # Expand dimensions to match the input shape of your model
    preprocessed_frame = np.expand_dims(normalized_frame, axis=-1)   
    return np.expand_dims(preprocessed_frame, axis=0)

def calculate_EAR(eye):  
    # calculate the vertical distances 
    y1 = dist.euclidean(eye[1], eye[5]) 
    y2 = dist.euclidean(eye[2], eye[4]) 
  
    # calculate the horizontal distance 
    x1 = dist.euclidean(eye[0], eye[3]) 
  
    # calculate the EAR 
    EAR = (y1+y2) / x1 
    return EAR 

def video_analysis(request):
    return render(request,"upload.html")

def scan_live_face(request):
    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    # Open the webcam (assuming the default camera index, 0)
    cap = cv2.VideoCapture(0)
    # Get the default frames per second (fps) of the camera
    fps = 16.2
    # Get the default frame size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like MJPG, DIVX, etc.
    output_filename=f'record_video_{uuid.uuid4()}.mp4'

    video_output = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    # Variables
    blink_thresh = 0.45
    succ_frame = 2
    count_frame = 0
    blinks_per_minute = 0
    current_second_start_time = time.time()
    frame_skip = 11
    frame_count = 0
    eye_contact = None
    hand_move = None
    eye_bling = None
    b_confidence = None
    thanks = None
    greet_gesture = None

    # Initialize variables for tracking time
    start_time = time.time()
    total_detected_time = 0
    total_not_detected_time = 0

    # Initialize variables for Body Posture
    good_posture_time = 0
    bad_posture_time = 0

    global stop_recording
    global audio_frames  # Use the global variable to store audio frames

    # Start recording audio in a separate thread
    audio_thread = threading.Thread(target=lambda: audio_frames.extend(record_audio()))
    audio_thread.start()

    while True:
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Null Frames")
            break
        video_output.write(image)
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames

        try:
            posture = detect_body_posture(image, fps)
            good_time, bad_time = posture
            print("Good Posture Time:", good_time)
            print("Bad Posture Time:", bad_time)
            if good_time > 0:
                good_posture_time += good_time
            else:
                bad_posture_time += bad_time

        except Exception as e:
            print(e)

        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame)

        if len(faces) > 0:
            total_detected_time += time.time() - start_time
            start_time = time.time()
            for face in faces:
                landmarks = landmark_predict(gray_frame, face)
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            total_not_detected_time += time.time() - start_time
            start_time = time.time()

        frame = cv2.flip(image, 1)

        # Emotion Changes Detection
        predicted_emotion = get_emotion_change(face_cascade,image)
        if predicted_emotion is not None:
            print("Emotion Changes:", predicted_emotion)
            cv2.putText(image, predicted_emotion, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            pass
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Hand Movement, Thanks Gesture, and body confidence detection
        greeting_gesture = hand_greeting_gesture(frame)
        hand_track = hand_movement(image)
        thanks_gesture = get_thanks_gesture(image)
        confidence = body_confidence(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if greeting_gesture == "Namaste" or greeting_gesture == "Hi/Hello":
            greet_gesture = "Greeting gesture included"
            cv2.putText(image, greeting_gesture, (20, 100), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
        else:
            pass

        if hand_track is not None:
            hand_move = 'Hand Moving'
            hand_track, x, y = hand_track
            cv2.putText(image, 'Hand Moving', (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            pass

        if thanks_gesture is not None:
            thanks_gesture, x, y = thanks_gesture
            thanks = "Thanking gesture included"
            cv2.putText(image, 'Thanks Gesture', (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        else:
            pass

        if confidence:
            b_confidence = "Confident body posture"
            cv2.putText(image, f"Posture: {confidence}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 153, 51), 2)
        else:
            pass

        # Eye Contact detection
        eye_distance, x, y, w, h = eye_contact_detection(image, face_cascade, eye_cascade)
        eye_contact_threshold = 20
        if eye_distance < eye_contact_threshold:
            eye_contact = 'Eye Contact'
            cv2.putText(image, 'Eye Contact', (x, y + h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 233, 51), 2)

        # Eye Blinging detection
        blinging_detected = eye_blinging(image)
        if blinging_detected < blink_thresh:
            count_frame += 1
        else:
            if count_frame >= succ_frame:
                blinks_per_minute += 1
                cv2.putText(image, 'Blink Detected', (60, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
            else:
                count_frame = 0

        elapsed_time = time.time() - current_second_start_time
        if elapsed_time >= 60:
            if blinks_per_minute > 12:
                eye_bling = "Blink more often"
                cv2.putText(image, f'{blinks_per_minute} Blinks in 1 Second', (60, 100), cv2.FONT_HERSHEY_DUPLEX, 1,
                            (0, 0, 255), 1)
            blinks_per_minute = 0
            current_second_start_time = time.time()

        # Smile detection
        smile_detect, x, y, w, h = smile_detection(image, face_cascade, smile_cascade)
        if smile_detect > 0:
            cv2.putText(image, 'Smiling', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the frame to the output video.
        video_output.write(image)
        new_width = 500
        new_height = 600

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height))

        # Display the frame.
        cv2.imshow('Video', resized_image)

        # Break the loop if 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_recording = True
            break

    # Release video capture and writer objects.
    cap.release()
    video_output.release()
    cv2.destroyAllWindows()
    # Stop recording audio
    stop_recording = True
    audio_thread.join()

    # Save the recorded audio
    save_audio(audio_frames, OUTPUT_FILE_PATH)
    print(f"Audio saved to {OUTPUT_FILE_PATH}")

    print(OUTPUT_FILE_PATH)
    language_analysis, voice_modulation,energy_category,filler_words_string,frequently_used_words,greeting_words = analyze_language_and_voice(OUTPUT_FILE_PATH)
    # Get speech rate
    speech_rate = calculate_speech_rate(OUTPUT_FILE_PATH)
    monotone = voice_monotone(OUTPUT_FILE_PATH)
    pauses = detect_voice_pauses(OUTPUT_FILE_PATH)
    print(f"Speech rate: {speech_rate:.2f} words/min")
    print("Language Analysis:", language_analysis)
    print("energy_level Analysis:", energy_category)
    print("filler_words Analysis:", filler_words_string)
    print("frequently_used_words Analysis:", frequently_used_words)
    print("Voice Modulation Analysis:", voice_modulation)
    print("Greeting Word Analysis:", greeting_words)
    print("Voice Tone Analysis:", monotone)
    print("Voice Pauses Analysis:", pauses)
    if len(greeting_words) > 0:
        greeting = "Greeting included"
    else:
        greeting = None

    emo = voice_emotion(OUTPUT_FILE_PATH)
    # Convert NumPy array to Python list
    voice_emo = emo.tolist() if isinstance(emo, np.ndarray) else emo
    
    try:
        video_output_file = merge_audio_video(output_filename,OUTPUT_FILE_PATH)
    except:
        video_output_file = None

    body_posture = None
    if good_posture_time > 0:
        print(good_posture_time)
        print(bad_posture_time)
        posture_ratio = good_posture_time/(good_posture_time + bad_posture_time)
        if posture_ratio > 0.5:
            body_posture = "Good Body Posture"
        else:
            body_posture = "Bad Body Posture"
    else:
        pass
    total_len = total_detected_time + total_not_detected_time

    # Check if total_len is zero before calculating the ratio
    if total_len == 0:
        # Handle the case where total_len is zero
        face_detected = "Unable to determine facial detection"
    else:
        ratio = total_detected_time / total_len
        if ratio > 0.5:
            face_detected = "Appropriate Facial Detected."
        else:
            face_detected = "Appropriate Facial Not Detected."

    analysed_score = get_analysis_score(speech_rate, filler_words_string, frequently_used_words, b_confidence, eye_bling, hand_move,
                       eye_contact, thanks, greeting, greet_gesture, monotone, pauses, face_detected, body_posture,voice_emo)
    video_data = None
    try:
        video_data = VideoRecognition.objects.get(id = video_output_file)
        video_data.analysis_score = round(analysed_score,2)
        video_data.language_analysis= language_analysis
        video_data.voice_modulation_analysis = voice_modulation
        video_data.energy_level_analysis= energy_category
        video_data.word_per_minute=round(speech_rate,2)
        video_data.filler_words_used=filler_words_string
        video_data.frequently_used_word=frequently_used_words
        video_data.voice_emotion = voice_emo
        video_data.confidence = b_confidence
        video_data.eye_bling = eye_bling
        video_data.hand_movement= hand_move
        video_data.eye_contact=eye_contact
        video_data.thanks_gesture=thanks
        video_data.greeting=greeting
        video_data.greeting_gesture=greet_gesture
        video_data.voice_tone = monotone
        video_data.voice_pauses=pauses
        video_data.appropriate_facial = face_detected
        video_data.body_posture=body_posture
        video_data.save()
    except Exception as e:
        print(e)
        pass
    try:
        print(output_filename)
        # os.remove(video_output_file)
        os.remove(output_filename)
        # os.remove(audio_file_path)
        os.remove(OUTPUT_FILE_PATH)
    except Exception as e:
        print(e)
        pass
    return redirect("analized_video_detail",video_data.id)


# Audio recording code start from here **************************************

CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
OUTPUT_FILE_PATH = f'speeches/record_audio_{uuid.uuid4()}.wav'
# OUTPUT_FILE_PATH = os.path.join(project_root_dir, f'speeches/record_audio_{uuid.uuid4()}.wav')

# Add a global variable to signal when to stop recording
stop_recording = False
# Define a global variable to store audio frames
audio_frames = []


def record_audio():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    print("Recording...")

    frames = []
    while not stop_recording:
        data = stream.read(CHUNK_SIZE)
        frames.append(data)

    print("Finished recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    return frames

def save_audio(frames, output_file):
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Audio recording code end **************************************
from django.core.files import File
from django.core.files.base import ContentFile
def merge_audio_video(video_filename, audio_filename):
    # Get the desired video title
    title = str(uuid.uuid4())
    # Open the video and audio
    video_clip = VideoFileClip(video_filename)
    audio_clip = AudioFileClip(audio_filename)
    # Set the audio of the video clip
    video_clip = video_clip.set_audio(audio_clip)
    # Export the final video with audio
    # output_filename = f"{title}.mp4"
    output_filename=f"{title}.mp4"
    video_clip.write_videofile(output_filename, codec='libx264', audio_codec='aac')
    # Create an instance of the Video model
    video = VideoRecognition()
    # Open the video file
    with open(output_filename, 'rb') as f:
        # Assign the video file to the model's FileField
        video.video_file.save(os.path.basename(output_filename), File(f), save=False)
    # Set the title of the video (you can adjust this based on your requirements)
    video.name = output_filename
    # Save the Video model
    video.save()
    os.remove(output_filename)
    return video.id
    # except Exception as e:
    #     print(f"Error: {e}")
    #     return None

def get_analysis_score(speech_rate, filler_words, words_list, b_confidence, eye_bling, hand_move,
                       eye_contact, thanks, greeting, greet_gesture, monotone, pauses, face_detected, body_posture,voice_emo):

    negative_point_data = 0

    if speech_rate == 0.0:
        negative_point_data += 1
    if not filler_words:
        negative_point_data += 1
    if not words_list:
        negative_point_data += 1
    if b_confidence == "Not confident body posture":
        negative_point_data += 1
    if eye_bling == "Blink more often":
        negative_point_data += 1
    if hand_move is None:
        negative_point_data += 1
    if eye_contact is None:
        negative_point_data += 1
    if thanks is None:
        negative_point_data += 1
    if greeting is None:
        negative_point_data += 1
    if greet_gesture is None:
        negative_point_data += 1
    if monotone == "Voice is monotone.":
        negative_point_data += 1
    if pauses == "Pauses seem unnatural":
        negative_point_data += 1
    if face_detected == "Appropriate Facial Not Detected.":
        negative_point_data += 1
    if body_posture is None or body_posture == "Bad Body Posture":
        negative_point_data += 1
    if not voice_emo:
        negative_point_data += 1

    print(negative_point_data)
    percentage = (negative_point_data / 15) * 100
    round_percentage = round(percentage, 2)
    score = 100 - round_percentage
    return score  # Invert the score to get a percentage


def eye_blinging(image):
    # Eye landmarks 
    (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
    (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'] 
    frame = imutils.resize(image, width=640) 
    # converting frame to gray scale to 
    # pass to detector 
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_faces = detector(img_gray) 
    for face in eye_faces:            
        # landmark detection 
        shape = landmark_predict(img_gray, face) 
        # converting the shape class directly 
        # to a list of (x,y) coordinates 
        shape = face_utils.shape_to_np(shape) 
        # parsing the landmarks list to extract 
        # lefteye and righteye landmarks--# 
        lefteye = shape[L_start: L_end] 
        righteye = shape[R_start:R_end] 
        # Calculate the EAR 
        left_EAR = calculate_EAR(lefteye) 
        right_EAR = calculate_EAR(righteye) 
        # Check if any EAR calculation is unsuccessful
        if left_EAR is None or right_EAR is None:
            return 0.0  # or any default value you prefer
        # Avg of left and right eye EAR 
        avg = (left_EAR + right_EAR) / 2
        return avg
    return 0.0  # Return default value if no face is found

def eye_contact_detection(image, face_cascade, eye_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    x, y, w, h = 0, 0, 0, 0  # Initialize these variables before the loop
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        
        # Eyes detection
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate
            eyes = sorted(eyes, key=lambda eye: eye[0])
            
            # Extract the coordinates of the two eyes
            (eye1_x, eye1_y, eye1_w, eye1_h), (eye2_x, eye2_y, eye2_w, eye2_h) = eyes[:2]
            
            # Calculate the center of each eye
            eye1_center = (eye1_x + eye1_w // 2, eye1_y + eye1_h // 2)
            eye2_center = (eye2_x + eye2_w // 2, eye2_y + eye2_h // 2)
            
            # Calculate the horizontal distance between the centers
            eye_distance = abs(eye1_center[0] - eye2_center[0])
            
            return eye_distance, x, y, w, h  # Return a tuple when eyes are detected
        
    return 0, x, y, w, h  # Return default value if no face is found

def smile_detection(image, face_cascade, smile_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    x, y, w, h = 0, 0, 0, 0  # Initialize these variables before the loop
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        
        # Smile detection
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)           
        return len(smiles), x, y, w, h
    
    return 0, x, y, w, h  # Return default value if no face is found, along with the last values of x, y, w, h

def hand_greeting_gesture(frame):
    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]
            return className


def detect_greeting_words(text):
  """Detects the greeting words "Hello", "Hi", "Hey", "Good morning", "Good afternoon", "Good evening", "How are you?", "How's it going?", "What's up?", "Nice to see you", "Long time no see", "It's good to see you again", "It's a pleasure to meet you", and "How can I help you?" in the text.

  Args:
    text: The text to search for greeting words in.

  Returns:
    A list of greeting words found in the text.
  """
  greeting_words_regex = re.compile(r'(?i)\b(hello|hi|hey|good morning|good afternoon|good evening|how are you|how\'s it going|what\'s up|nice to see you|long time no see|it\'s good to see you again|it\'s a pleasure to meet you|how can I help you|namaskar|namastey|pranam|sat shri akal)\b')


  greeting_words = []
  for match in greeting_words_regex.finditer(text):
    greeting_words.append(match.group())
  return greeting_words

# load trained model for emotion detection
emotion_model = load_model("best_model.h5")

def get_emotion_change(face_cascade,image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray_img, 1.32, 5)
    for (x, y, w, h) in faces_detected:
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)  # convert to RGB
        roi_rgb = cv2.resize(roi_rgb, (224, 224))  # resize to (224, 224)
        img_pixels = np.expand_dims(roi_rgb, axis=0)
        img_pixels = img_pixels / 255.0  # Normalize pixel values to [0, 1]

        predictions = emotion_model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        return predicted_emotion
    return None


def detect_voice_pauses(audio_file_path):
    # Load the audio file
    audio = AudioSegment.from_file(audio_file_path)

    # Set the silence threshold for voice activity detection
    silence_threshold = -40  # Adjust this threshold based on your audio characteristics

    # Split the audio on silence to get segments with voice activity
    voice_segments = split_on_silence(audio, silence_thresh=silence_threshold)

    # Calculate total voice duration without pauses
    total_voice_duration_without_pauses = sum(len(segment) for segment in voice_segments)

    # Calculate total voice duration with pauses
    total_voice_duration_with_pauses = len(audio)

    # Check if total_voice_duration_with_pauses is zero before calculating the ratio
    if total_voice_duration_with_pauses == 0:
        # Handle the case where total_voice_duration_with_pauses is zero
        return "Unable to detect pauses"

    # Calculate the ratio of voice duration without pauses to total voice duration
    ratio = total_voice_duration_without_pauses / total_voice_duration_with_pauses

    if 0.5 < ratio < 0.9:
        return "Pauses seem natural"
    else:
        return "Pauses seem unnatural"



def voice_emotion(audio_file_path):
    # Load the model
    filename = 'modelForPrediction1.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    # Extract features from the trimmed audio file
    new_feature = extract_feature(audio_file_path, mfcc=True, chroma=True, mel=True)

    if new_feature is not None:
        new_feature = new_feature.reshape(1, -1)
        prediction = loaded_model.predict(new_feature)
        return prediction
    else:
        print("Error extracting features for prediction.")



def analyze_language_and_voice(audio_file_path):
    # Transcribe spoken words
    transcribed_text = transcribe_audio(audio_file_path)
    frequently_used_words = get_frequently_used_words(transcribed_text)
    words_list = []
    for word, frequency in frequently_used_words:
        words_list.append(word)

    # Analyze language characteristics (e.g., sentiment)
    language_analysis = analyze_language(transcribed_text)

    #Detect Greeting in voice
    greeting_words = detect_greeting_words(transcribed_text)

    # Analyze voice energy level
    audio = AudioSegment.from_wav(audio_file_path)
    energy_level = calculate_energy_level(audio)
    # Categorize energy level
    energy_category = categorize_energy_level(energy_level)
    filler_words = analyze_filler_words(transcribed_text)
    # Analyze voice modulation
    voice_modulation = analyze_voice_modulation(audio_file_path)
    return language_analysis, voice_modulation,energy_category,filler_words,words_list,greeting_words

def hand_movement(image):
    x = 0
    y = 0
    # Process the image with MediaPipe Hands
    hands_results = hands.process(image)
    # Check if hands are detected
    if hands_results.multi_hand_landmarks:
        return ("Hand Moving" , x, y)
    return None




def get_thanks_gesture(image):
    x = 0
    y = 0
    hands_results = hands.process(image)
    # Check if hands are detected
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]              
            # Define your logic for detecting a "thanks" gesture
            if thumb_tip.y < index_tip.y and middle_tip.y < index_tip.y:
                return ('Thanks Gesture', x, y)
            # Draw hand landmarks on the image
            # for lm in hand_landmarks.landmark:
            #     x, y = int(lm.x * w), int(lm.y * h)
            #     cv2.circle(image, (x, y), 7, (11, 80, 60), -1)
        return None
    return None


def body_confidence(image):
    pose_results = pose.process(image)
    if pose_results.pose_landmarks:
        # Access landmarks and analyze body posture
        # You can define your own rules for confident body posture
        # For simplicity, let's consider if shoulders are aligned
        left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if left_shoulder.x < right_shoulder.x:
            posture = "Confident"
        else:
            posture = "Not Confident"
        return posture

def get_frequently_used_words(transcribed_text):
    if transcribed_text == "Transcription could not be performed":
        most_common_words = []
        return most_common_words
    # Tokenize the transcribed text into words
    words = word_tokenize(transcribed_text)
    
    # Calculate the frequency distribution of words
    freq_dist = FreqDist(words)
    # Get the most common words
    most_common_words = freq_dist.most_common(10)  # You can adjust the number based on your preference

    return most_common_words

def analyze_filler_words(transcribed_text):
    # Define a list of common filler words
    filler_words = ["um", "uh", "like", "you know", "so","very", "actually", "basically", "literally", "well", "uhm", "uhh", "okay", "right", "I mean", "sort of", "kind of", "definitely", "obviously", "seriously", "totally", "absolutely", "basically", "essentially", "apparently", "apparently", "frankly", "honestly", "clearly", "you see", "mind you", "anyway", "however", "meanwhile", "nevertheless", "otherwise", "somehow", "therefore", "anyhow", "consequently", "furthermore", "otherwise", "moreover"]

    # Convert text to lowercase for case-insensitive matching
    transcribed_text_lower = transcribed_text.lower()

    used_filler_words = list(set(word for word in filler_words if word in transcribed_text_lower))

    return used_filler_words

def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)

    try:
        transcribed_text = recognizer.recognize_google(audio_data)
        return transcribed_text
    except sr.UnknownValueError:
        return "Transcription could not be performed"

def analyze_language(text):
    # Initialize the SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    # Get sentiment scores
    sentiment_scores = sid.polarity_scores(text)

    # Determine sentiment based on the compound score
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return {
        "sentiment": sentiment,
        # "sentiment_scores": sentiment_scores
    }

def voice_monotone(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    samples = np.array(audio.get_array_of_samples())
    # Calculate the root mean square (RMS) of the audio signal
    rms = np.sqrt(np.mean(np.square(samples)))

    # Print the RMS value for reference
    # print("RMS:", rms)

    # Set a threshold for determining monotone or clear voice
    min_threshold = 20  # Adjust this threshold based on your observations
    max_threshold = 50

    # Check if the RMS value is below the threshold
    if min_threshold < rms < max_threshold:
        return "Voice is clear."
    else:
        return "Voice is monotone."

def analyze_voice_modulation(audio_file_path):
    audio = AudioSegment.from_wav(audio_file_path)

    # Perform voice modulation analysis (add your logic here)

    pitch = audio.dBFS

    # Check if pitch is -inf and handle it
    if pitch == float('-inf'):
        pitch = None

    # You can define your own criteria for rating voice modulation
    if pitch is not None:
        if pitch > -12:
            modulation_rating = "Excellent"
        elif -12 >= pitch >= -25:
            modulation_rating = "Good"
        else:
            modulation_rating = "Needs Improvement"
    else:
        modulation_rating = "Not Available"

    return {
        # "duration_seconds": len(audio) / 1000,  # Duration in seconds
        "pitch": round(pitch,2),  # Pitch in dB
        "modulation_rating": modulation_rating,
        # Add more voice modulation characteristics as needed
    }
    

def calculate_energy_level(audio):
    # Calculate the root mean square (RMS) to estimate energy level
    rms = np.sqrt(np.mean(np.square(audio.get_array_of_samples())))
    
    # Adjust the range as needed
    min_rms = 0.0  # Set the minimum expected RMS value
    max_rms = 100.0  # Set the maximum expected RMS value
    
    # Normalize the RMS value to the range [0, 1]
    normalized_energy = (rms - min_rms) / (max_rms - min_rms)
    return normalized_energy

def categorize_energy_level(energy_level):
    # Define thresholds for categorization
    low_threshold = 0.3
    high_threshold = 0.7

    if energy_level < low_threshold:
        return "Low"
    elif low_threshold <= energy_level < high_threshold:
        return "Medium"
    else:
        return "High"

def generate_audio_file(video_path):
    # Speech recognition setup
    recognizer = sr.Recognizer()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    audio_file_path = f"speeches/file_{timestamp}.wav"  # Use a timestamp to create a unique audio file name

    # Use ffmpeg to extract audio from the video
    ffmpeg_path = "/usr/bin/ffmpeg"  # Replace with the actual path
    video_to_audio_command = [ffmpeg_path, "-i", video_path, "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", audio_file_path]

    try:
        subprocess.run(video_to_audio_command, check=True)
        print("Audio file generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    
    return audio_file_path

def calculate_speech_rate(audio_file_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio_data)
        word_count = len(text.split())
    except sr.UnknownValueError:
        word_count = 0

    # Use the sample rate from AudioFile object
    frame_rate = source.SAMPLE_RATE

    # Assume speech duration is the length of the audio in seconds
    speech_duration = len(audio_data.frame_data) / frame_rate

    # Check if speech_duration is zero before calculating speech rate
    if speech_duration == 0:
        # Handle the case where speech_duration is zero (e.g., empty audio file)
        speech_rate = 0
    else:
        # Calculate speech rate in words per minute
        speech_rate = (word_count / speech_duration) * 60

    return speech_rate


def analized_video_detail(request,video_id):
    try:
        video_data = VideoRecognition.objects.get(id=video_id)
    except:
        video_data=None
    return render(request,"video_detail.html",{
        "video_data":video_data
    })

def analized_video_list(request):
    all_data = VideoRecognition.objects.all()
    return render(request,"analize_video_list.html",{
        "all_data":all_data
    })

    

# API For Live Video Analysis ###############################################################

class LiveAnalysisView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    serializer_class = VideoSerializer
    
    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            analysez_data = analyse_live()
            try:
                data = VideoRecognition.objects.get(id = analysez_data)
                serializer = VideoDataSerializer(data)  # Use your VideoDataSerializer to serialize the instance
                serialized_data = serializer.data
            except:
                serialized_data = None
            return Response(
                serialized_data,
                status=status.HTTP_201_CREATED
            )
        
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )
    
def analyse_live():
    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')


    # Open the webcam (assuming the default camera index, 0)
    cap = cv2.VideoCapture(0)
    # Get the default frames per second (fps) of the camera
    fps = 16.2
    # Get the default frame size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like MJPG, DIVX, etc.
    output_filename=f'record_video_{uuid.uuid4()}.mp4'
    video_output = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    # Variables
    blink_thresh = 0.45
    succ_frame = 2
    count_frame = 0
    blinks_per_minute = 0
    current_second_start_time = time.time()
    frame_skip = 11
    frame_count = 0
    eye_contact = None
    hand_move = None
    eye_bling = None
    b_confidence = None
    thanks = None
    greet_gesture = None

    # Initialize variables for tracking time
    start_time = time.time()
    total_detected_time = 0
    total_not_detected_time = 0

    # Initialize variables for Body Posture
    good_posture_time = 0
    bad_posture_time = 0

    global stop_recording
    global audio_frames  # Use the global variable to store audio frames

    # Start recording audio in a separate thread
    audio_thread = threading.Thread(target=lambda: audio_frames.extend(record_audio()))
    audio_thread.start()

    while True:
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Null Frames")
            break
        video_output.write(image)
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames

        try:
            posture = detect_body_posture(image, fps)
            good_time, bad_time = posture
            print("Good Posture Time:", good_time)
            print("Bad Posture Time:", bad_time)
            if good_time > 0:
                good_posture_time += good_time
            else:
                bad_posture_time += bad_time

        except Exception as e:
            print(e)

        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame)

        if len(faces) > 0:
            total_detected_time += time.time() - start_time
            start_time = time.time()
            for face in faces:
                landmarks = landmark_predict(gray_frame, face)
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            total_not_detected_time += time.time() - start_time
            start_time = time.time()

        frame = cv2.flip(image, 1)

        # Emotion Changes Detection
        predicted_emotion = get_emotion_change(face_cascade,image)
        if predicted_emotion is not None:
            print("Emotion Changes:", predicted_emotion)
            cv2.putText(image, predicted_emotion, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            pass
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Hand Movement, Thanks Gesture, and body confidence detection
        greeting_gesture = hand_greeting_gesture(frame)
        hand_track = hand_movement(image)
        thanks_gesture = get_thanks_gesture(image)
        confidence = body_confidence(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if greeting_gesture == "Namaste" or greeting_gesture == "Hi/Hello":
            greet_gesture = "Greeting gesture included"
            cv2.putText(image, greeting_gesture, (20, 100), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
        else:
            pass

        if hand_track is not None:
            hand_move = 'Hand Moving'
            hand_track, x, y = hand_track
            cv2.putText(image, 'Hand Moving', (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            pass

        if thanks_gesture is not None:
            thanks_gesture, x, y = thanks_gesture
            thanks = "Thanking gesture included"
            cv2.putText(image, 'Thanks Gesture', (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        else:
            pass

        if confidence:
            b_confidence = "Confident body posture"
            cv2.putText(image, f"Posture: {confidence}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 153, 51), 2)
        else:
            pass

        # Eye Contact detection
        eye_distance, x, y, w, h = eye_contact_detection(image, face_cascade, eye_cascade)
        eye_contact_threshold = 20
        if eye_distance < eye_contact_threshold:
            eye_contact = 'Eye Contact'
            cv2.putText(image, 'Eye Contact', (x, y + h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 233, 51), 2)

        # Eye Blinging detection
        blinging_detected = eye_blinging(image)
        if blinging_detected < blink_thresh:
            count_frame += 1
        else:
            if count_frame >= succ_frame:
                blinks_per_minute += 1
                cv2.putText(image, 'Blink Detected', (60, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
            else:
                count_frame = 0

        elapsed_time = time.time() - current_second_start_time
        if elapsed_time >= 60:
            if blinks_per_minute > 12:
                eye_bling = "Blink more often"
                cv2.putText(image, f'{blinks_per_minute} Blinks in 1 Second', (60, 100), cv2.FONT_HERSHEY_DUPLEX, 1,
                            (0, 0, 255), 1)
            blinks_per_minute = 0
            current_second_start_time = time.time()

        # Smile detection
        smile_detect, x, y, w, h = smile_detection(image, face_cascade, smile_cascade)
        if smile_detect > 0:
            cv2.putText(image, 'Smiling', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the frame to the output video.
        video_output.write(image)
        new_width = 500
        new_height = 600

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height))

        # Display the frame.
        cv2.imshow('Video', resized_image)

        # Break the loop if 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_recording = True
            break

    # Release video capture and writer objects.
    cap.release()
    video_output.release()
    cv2.destroyAllWindows()
    # Stop recording audio
    stop_recording = True
    audio_thread.join()

    # Save the recorded audio
    save_audio(audio_frames, OUTPUT_FILE_PATH)
    print(f"Audio saved to {OUTPUT_FILE_PATH}")

    print(OUTPUT_FILE_PATH)
    language_analysis, voice_modulation,energy_category,filler_words,words_list,greeting_words = analyze_language_and_voice(OUTPUT_FILE_PATH)
    # Get speech rate
    speech_rate = calculate_speech_rate(OUTPUT_FILE_PATH)
    monotone = voice_monotone(OUTPUT_FILE_PATH)
    pauses = detect_voice_pauses(OUTPUT_FILE_PATH)
    print(f"Speech rate: {speech_rate:.2f} words/min")
    print("Language Analysis:", language_analysis)
    print("energy_level Analysis:", energy_category)
    print("filler_words Analysis:", filler_words)
    print("frequently_used_words Analysis:", words_list)
    print("Voice Modulation Analysis:", voice_modulation)
    print("Greeting Word Analysis:", greeting_words)
    print("Voice Tone Analysis:", monotone)
    print("Voice Pauses Analysis:", pauses)
    if len(greeting_words) > 0:
        greeting = "Greeting included"
    else:
        greeting = None
    frequently_used_words = json.dumps(words_list)
    filler_words_string = json.dumps(filler_words)
    emo = voice_emotion(OUTPUT_FILE_PATH)
    # Convert NumPy array to Python list
    emo_list = emo.tolist() if isinstance(emo, np.ndarray) else emo
    # Convert to JSON
    voice_emo = json.dumps(emo_list)
    try:
        video_output_file = merge_audio_video(output_filename,OUTPUT_FILE_PATH)
    except:
        video_output_file = None

    body_posture = None
    if good_posture_time > 0:
        print(good_posture_time)
        print(bad_posture_time)
        posture_ratio = good_posture_time/(good_posture_time + bad_posture_time)
        if posture_ratio > 0.5:
            body_posture = "Good Body Posture"
        else:
            body_posture = "Bad Body Posture"
    else:
        pass
    total_len = total_detected_time + total_not_detected_time

    # Check if total_len is zero before calculating the ratio
    if total_len == 0:
        # Handle the case where total_len is zero
        face_detected = "Unable to determine facial detection"
    else:
        ratio = total_detected_time / total_len
        if ratio > 0.5:
            face_detected = "Appropriate Facial Detected."
        else:
            face_detected = "Appropriate Facial Not Detected."
    
    video_data = None
    try:
        video_data = VideoRecognition.objects.get(id = video_output_file)
        video_data.language_analysis= language_analysis
        video_data.voice_modulation_analysis = voice_modulation
        video_data.energy_level_analysis= energy_category
        video_data.word_per_minute=speech_rate
        video_data.filler_words_used=filler_words_string
        video_data.frequently_used_word=frequently_used_words
        video_data.voice_emotion = voice_emo
        video_data.confidence = b_confidence
        video_data.eye_bling = eye_bling
        video_data.hand_movement= hand_move
        video_data.eye_contact=eye_contact
        video_data.thanks_gesture=thanks
        video_data.greeting=greeting
        video_data.greeting_gesture=greet_gesture
        video_data.voice_tone = monotone
        video_data.voice_pauses=pauses
        video_data.appropriate_facial = face_detected
        video_data.body_posture=body_posture
        video_data.save()
        data_id = video_data.id
    except Exception as e:
        print(e)
        data_id = None
    try:
        os.remove(output_filename)
        os.remove(OUTPUT_FILE_PATH)
    except:
        pass
    return data_id


# Analysed video list api code **********************************
class AnalysedVideoListView(APIView):
    def get(self,request):
        all_data = VideoRecognition.objects.all()
        serializer = VideoDataListSerializer(all_data, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


