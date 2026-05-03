import cv2
from deepface import DeepFace
import pandas as pd

class PersonalityPredictor:
    def __init__(self):
        self.emotion_history = []
        self.personality_mapping = {
            'Openness': {'happy': 0.4, 'surprise': 0.6},
            'Conscientiousness': {'neutral': 0.8, 'happy': 0.2},
            'Extraversion': {'happy': 0.7, 'surprise': 0.3},
            'Agreeableness': {'happy': 0.6, 'neutral': 0.4},
            'Neuroticism': {'sad': 0.5, 'angry': 0.3, 'fear': 0.2}
        }

    def analyze_frame(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            dominant_emotion = result[0]['dominant_emotion']
            self.emotion_history.append(dominant_emotion)
            return dominant_emotion, result[0]['region']
        except:
            return None, None

    def calculate_personality_scores(self):
        if not self.emotion_history: return None
        emotion_counts = pd.Series(self.emotion_history).value_counts(normalize=True).to_dict()
        scores = {trait: sum(emotion_counts.get(emo, 0) * w for emo, w in ws.items()) 
                  for trait, ws in self.personality_mapping.items()}
        total = sum(scores.values())
        return {k: (v/total)*100 for k, v in scores.items()} if total > 0 else scores

    def reset(self):
        self.emotion_history = []