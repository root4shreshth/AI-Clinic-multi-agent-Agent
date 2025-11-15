"""
Agent 1: Voice Intake Specialist
Handles audio input and converts to clean transcripts with urgency detection
"""

import os
import numpy as np
from google.cloud import speech_v1p1beta1 as speech
from typing import Dict, Optional
import librosa
from dataclasses import dataclass


@dataclass
class VoiceIntakeResult:
    transcript: str
    confidence: float
    urgency_level: str  # "low", "medium", "high"
    urgency_score: float
    speaker_emotions: Dict[str, float]
    duration_seconds: float


class VoiceIntakeAgent:
    """
    Voice Intake Specialist Agent using Google Speech-to-Text
    with medical conversation optimization and urgency detection
    """
    
    def __init__(self, language_code: str = "en-US"):
        self.client = speech.SpeechClient()
        self.language_code = language_code
        
        # Medical-specific vocabulary for better recognition
        self.medical_phrases = [
            "chest pain", "shortness of breath", "fever", "nausea",
            "dizziness", "headache", "abdominal pain", "cough",
            "fatigue", "allergies", "medications", "hypertension",
            "diabetes", "asthma", "heart disease"
        ]
    
    def process_audio(self, audio_file_path: str) -> VoiceIntakeResult:
        """
        Main processing pipeline for voice intake
        
        Args:
            audio_file_path: Path to audio file (WAV, FLAC, MP3)
            
        Returns:
            VoiceIntakeResult with transcript and analysis
        """
        # Step 1: Load and preprocess audio
        audio_data = self._load_audio(audio_file_path)
        
        # Step 2: Analyze urgency from audio features
        urgency_analysis = self._analyze_urgency(audio_data, audio_file_path)
        
        # Step 3: Transcribe with Google Speech-to-Text
        transcript_result = self._transcribe_audio(audio_file_path)
        
        # Step 4: Combine results
        return VoiceIntakeResult(
            transcript=transcript_result["transcript"],
            confidence=transcript_result["confidence"],
            urgency_level=urgency_analysis["level"],
            urgency_score=urgency_analysis["score"],
            speaker_emotions=urgency_analysis["emotions"],
            duration_seconds=urgency_analysis["duration"]
        )
    
    def _load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file and normalize"""
        audio, sr = librosa.load(file_path, sr=16000)
        return audio
    
    def _analyze_urgency(self, audio_data: np.ndarray, file_path: str) -> Dict:
        """
        Analyze audio features to detect urgency
        - Voice pitch (higher pitch = more stress)
        - Speech rate (faster = more urgent)
        - Energy/loudness variations
        - Voice tremor
        """
        duration = len(audio_data) / 16000.0
        
        # Extract features
        pitch = librosa.yin(audio_data, fmin=50, fmax=400)
        pitch_mean = np.nanmean(pitch)
        pitch_std = np.nanstd(pitch)
        
        # Speech rate (zero crossing rate as proxy)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        speech_rate = np.mean(zcr)
        
        # Energy
        rms = librosa.feature.rms(y=audio_data)[0]
        energy_std = np.std(rms)
        
        # Calculate urgency score (0-1)
        urgency_score = 0.0
        emotions = {
            "stress": 0.0,
            "pain": 0.0,
            "anxiety": 0.0,
            "calm": 0.0
        }
        
        # High pitch variation indicates stress
        if pitch_std > 30:
            urgency_score += 0.3
            emotions["stress"] = min(1.0, pitch_std / 50)
        
        # Fast speech rate indicates urgency
        if speech_rate > 0.15:
            urgency_score += 0.3
            emotions["anxiety"] = min(1.0, speech_rate / 0.2)
        
        # High energy variation indicates distress
        if energy_std > 0.05:
            urgency_score += 0.2
            emotions["pain"] = min(1.0, energy_std / 0.1)
        
        # Low variation = calm
        if pitch_std < 15 and speech_rate < 0.1:
            emotions["calm"] = 0.8
        
        urgency_score = min(1.0, urgency_score)
        
        # Determine urgency level
        if urgency_score > 0.7:
            level = "high"
        elif urgency_score > 0.4:
            level = "medium"
        else:
            level = "low"
        
        return {
            "level": level,
            "score": urgency_score,
            "emotions": emotions,
            "duration": duration
        }
    
    def _transcribe_audio(self, file_path: str) -> Dict:
        """
        Transcribe audio using Google Speech-to-Text
        with medical conversation optimization
        """
        with open(file_path, "rb") as audio_file:
            content = audio_file.read()
        
        audio = speech.RecognitionAudio(content=content)
        
        # Configure recognition with medical enhancements
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=self.language_code,
            
            # Enable advanced features
            enable_automatic_punctuation=True,
            enable_speaker_diarization=True,
            diarization_speaker_count=2,  # Patient + interviewer
            
            # Medical vocabulary boost
            speech_contexts=[
                speech.SpeechContext(phrases=self.medical_phrases)
            ],
            
            # Use medical conversation model
            model="medical_conversation",
            use_enhanced=True
        )
        
        # Perform transcription
        response = self.client.recognize(config=config, audio=audio)
        
        # Process results
        transcript = ""
        confidence_scores = []
        
        for result in response.results:
            alternative = result.alternatives[0]
            transcript += alternative.transcript + " "
            confidence_scores.append(alternative.confidence)
        
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return {
            "transcript": transcript.strip(),
            "confidence": avg_confidence
        }
    
    def process_streaming_audio(self, audio_stream):
        """
        Process real-time audio stream (for live demo)
        This would be used for phone/web interface
        """
        # Implementation for streaming would go here
        # Using streaming_recognize() from Google Speech API
        pass


# Demo usage
if __name__ == "__main__":
    agent = VoiceIntakeAgent()
    
    # Test with sample audio
    result = agent.process_audio("sample_patient_audio.wav")
    
    print(f"Transcript: {result.transcript}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Urgency: {result.urgency_level} ({result.urgency_score:.2f})")
    print(f"Emotions: {result.speaker_emotions}")