from gtts import gTTS
from googletrans import Translator
import os

# The text that you want to convert to audio in Hindi
hindi_text = 'पूजा स्थल अधिनियम की संवैधानिकता, राम मंदिर के बाद काशी और मथुरा की तैयारी'

# Language codes for translation
source_language = 'hi'
target_language = 'en'

# Translate the Hindi text to English
translator = Translator()
english_text = translator.translate(hindi_text, src=source_language, dest=target_language).text

# Passing the translated text and language to the gTTS engine
english_obj = gTTS(text=english_text, lang=target_language, slow=False)

# Saving the converted audio in an mp3 file named hindi_to_english
english_obj.save("hindi_to_english.wav")

