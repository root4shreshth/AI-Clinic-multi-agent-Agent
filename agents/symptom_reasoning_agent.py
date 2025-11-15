"""
Agent 2: Symptom Reasoning Engine
Powered by Gemini 2.0 - extracts symptoms and asks clarifying questions
"""

import os
import google.generativeai as genai
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class SymptomSeverity(Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class ExtractedSymptom:
    name: str
    severity: SymptomSeverity
    duration: str  # e.g., "2 days", "3 hours"
    location: Optional[str]  # e.g., "chest", "abdomen"
    characteristics: List[str]  # e.g., ["sharp", "radiating"]
    triggers: List[str]  # What makes it worse
    relievers: List[str]  # What makes it better


@dataclass
class SymptomAnalysisResult:
    symptoms: List[ExtractedSymptom]
    medical_history: List[str]
    current_medications: List[str]
    allergies: List[str]
    clarifying_questions: List[str]
    missing_critical_info: List[str]
    confidence_score: float


class SymptomReasoningAgent:
    """
    Symptom Reasoning Engine using Gemini 2.0 Flash
    Extracts structured symptom data and generates intelligent follow-up questions
    """
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        # Use Gemini 2.0 Flash for fast, efficient reasoning
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Medical reasoning prompts
        self.system_prompt = """You are an expert medical AI assistant specializing in patient intake and symptom analysis. Your role is to:

1. Extract all symptoms mentioned with complete details (severity, duration, location, characteristics)
2. Identify medical history, medications, and allergies
3. Recognize missing critical information based on the symptoms described
4. Generate focused, medically-relevant clarifying questions
5. Flag any red flag symptoms that indicate high urgency

Output your analysis in a structured JSON format.

Critical thinking rules:
- Chest pain → ask about radiation, cardiac risk factors
- Abdominal pain → ask about onset, eating relationship
- Headache → ask about sudden onset, worst headache ever
- Shortness of breath → ask about onset, exertion relationship
- Fever → ask about duration, associated symptoms

Always consider:
- Age-related risk factors
- Symptom combinations that suggest serious conditions
- Timeline and progression of symptoms"""
    
    def analyze_symptoms(self, transcript: str, urgency_context: Dict = None) -> SymptomAnalysisResult:
        """
        Main symptom analysis pipeline
        
        Args:
            transcript: Patient's voice transcript
            urgency_context: Additional context from voice analysis
            
        Returns:
            SymptomAnalysisResult with structured data
        """
        
        # Build the analysis prompt
        prompt = self._build_analysis_prompt(transcript, urgency_context)
        genai.configure()
        # Call Gemini for reasoning
        response = self.model.generate_content(prompt)
        
        # Parse structured output
        analysis = self._parse_gemini_response(response.text)
        
        return analysis
    
    def _build_analysis_prompt(self, transcript: str, urgency_context: Dict = None) -> str:
        """Build comprehensive prompt for Gemini"""
        
        urgency_info = ""
        if urgency_context:
            urgency_info = f"""
Voice Analysis Context:
- Urgency Level: {urgency_context.get('urgency_level', 'unknown')}
- Detected Emotions: {urgency_context.get('emotions', {})}
- Speech Duration: {urgency_context.get('duration', 0)} seconds
"""
        
        prompt = f"""{self.system_prompt}

{urgency_info}

Patient Transcript:
"{transcript}"

Please provide a comprehensive analysis in the following JSON structure:
{{
    "symptoms": [
        {{
            "name": "symptom name",
            "severity": "mild/moderate/severe/critical",
            "duration": "time period",
            "location": "body location if applicable",
            "characteristics": ["descriptive terms"],
            "triggers": ["what makes it worse"],
            "relievers": ["what makes it better"]
        }}
    ],
    "medical_history": ["relevant conditions"],
    "current_medications": ["medication names"],
    "allergies": ["known allergies"],
    "clarifying_questions": [
        "Specific question based on symptoms"
    ],
    "missing_critical_info": [
        "What critical information is missing"
    ],
    "red_flags": [
        "Any concerning symptoms requiring immediate attention"
    ],
    "confidence_score": 0.95
}}

Be thorough but concise. Focus on medically relevant information."""
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> SymptomAnalysisResult:
        """Parse Gemini's JSON response into structured data"""
        import json
        import re
        
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            data = {
                "symptoms": [],
                "medical_history": [],
                "current_medications": [],
                "allergies": [],
                "clarifying_questions": [],
                "missing_critical_info": [],
                "confidence_score": 0.5
            }
        
        # Convert to structured objects
        symptoms = []
        for symp_data in data.get("symptoms", []):
            try:
                severity = SymptomSeverity(symp_data.get("severity", "moderate"))
            except ValueError:
                severity = SymptomSeverity.MODERATE
            
            symptom = ExtractedSymptom(
                name=symp_data.get("name", ""),
                severity=severity,
                duration=symp_data.get("duration", "unknown"),
                location=symp_data.get("location"),
                characteristics=symp_data.get("characteristics", []),
                triggers=symp_data.get("triggers", []),
                relievers=symp_data.get("relievers", [])
            )
            symptoms.append(symptom)
        
        return SymptomAnalysisResult(
            symptoms=symptoms,
            medical_history=data.get("medical_history", []),
            current_medications=data.get("current_medications", []),
            allergies=data.get("allergies", []),
            clarifying_questions=data.get("clarifying_questions", []),
            missing_critical_info=data.get("missing_critical_info", []),
            confidence_score=data.get("confidence_score", 0.8)
        )
    
    def generate_follow_up_questions(
        self,
        symptoms: List[ExtractedSymptom],
        context: Dict
    ) -> List[str]:
        """
        Generate intelligent follow-up questions based on symptom combinations
        This is the "auto-clarification" feature
        """
        
        prompt = f"""Based on these symptoms, generate 3-5 focused clarifying questions:

Symptoms: {[s.name for s in symptoms]}

Context:
{context}

Generate questions that:
1. Fill critical information gaps
2. Rule out serious conditions
3. Are specific and actionable
4. Use patient-friendly language

Return only the questions as a JSON array."""
        
        response = self.model.generate_content(prompt)
        
        # Parse questions
        import json
        try:
            questions = json.loads(response.text)
            return questions if isinstance(questions, list) else []
        except:
            return []


# Demo usage
if __name__ == "__main__":
    agent = SymptomReasoningAgent()
    
    sample_transcript = """
    I've had this chest pain for about 2 days now. It's like a tightness, 
    maybe 6 out of 10 in severity. I also have a cough and some fever. 
    I have high blood pressure and I take lisinopril daily. 
    I'm allergic to penicillin.
    """
    
    urgency_context = {
        "urgency_level": "high",
        "emotions": {"stress": 0.7, "pain": 0.6},
        "duration": 45.0
    }
    
    result = agent.analyze_symptoms(sample_transcript, urgency_context)
    
    print(f"Extracted {len(result.symptoms)} symptoms:")
    for symptom in result.symptoms:
        print(f"  - {symptom.name}: {symptom.severity.value} for {symptom.duration}")
    
    print(f"\nMedical History: {result.medical_history}")
    print(f"Medications: {result.current_medications}")
    print(f"Allergies: {result.allergies}")
    
    print(f"\nClarifying Questions:")
    for q in result.clarifying_questions:
        print(f"  - {q}")
    
    print(f"\nConfidence: {result.confidence_score:.2%}")