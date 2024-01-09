# def convert_hindi_text_to_english_audio(voice_text):
#     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#     audio_file_path = f"speeches/translate_{timestamp}.wav"
#     # Language codes for translation
#     source_language = 'hi'
#     target_language = 'en'

#     # Translate the Hindi text to English
#     translator = Translator()
#     english_text = translator.translate(voice_text, src=source_language, dest=target_language).text

#     # Passing the translated text and language to the gTTS engine
#     english_obj = gTTS(text=english_text, lang=target_language, slow=False)

#     # Saving the converted audio in a WAV file
#     english_obj.save(audio_file_path)

#     audio_file = f"speeches/translate_audio.wav"

#     # Use FFmpeg to overwrite the WAV file
#     ffmpeg_path = "/usr/bin/ffmpeg"  # Replace with the actual path
#     video_to_audio_command = [ffmpeg_path, '-y', '-i', audio_file_path, audio_file]

#     try:
#         subprocess.run(video_to_audio_command, check=True)
#         print("Audio file conversion successful.")
#     except subprocess.CalledProcessError as e:
#         print(f"Error: {e}")
#     os.remove(audio_file_path)
#     return audio_file