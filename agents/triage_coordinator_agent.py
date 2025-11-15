"""
Agent 4: Clinical Triage Coordinator (The Brain)
Combines all data sources and makes intelligent clinical decisions
"""

import os
import google.generativeai as genai
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class UrgencyLevel(Enum):
    IMMEDIATE = "immediate"  # Life-threatening, needs ER now
    URGENT = "urgent"  # Needs care within hours
    SEMI_URGENT = "semi_urgent"  # Needs care within 24 hours
    NON_URGENT = "non_urgent"  # Can wait for routine appointment


class DispositionType(Enum):
    EMERGENCY_DEPARTMENT = "emergency_department"
    URGENT_CARE = "urgent_care"
    PRIMARY_CARE_SAME_DAY = "primary_care_same_day"
    PRIMARY_CARE_ROUTINE = "primary_care_routine"
    TELEHEALTH = "telehealth"
    SELF_CARE = "self_care"


@dataclass
class RedFlag:
    symptom: str
    reasoning: str
    severity: str


@dataclass
class ClinicalContradiction:
    finding: str
    conflict: str
    recommendation: str


@dataclass
class TriageDecision:
    urgency_level: UrgencyLevel
    urgency_confidence: float
    disposition: DispositionType
    red_flags: List[RedFlag]
    contradictions: List[ClinicalContradiction]
    clinical_reasoning: str
    recommended_tests: List[str]
    specialist_referral: Optional[str]
    estimated_wait_time: str


class TriageCoordinatorAgent:
    """
    Clinical Triage Coordinator - The Brain of the system
    Synthesizes all agent inputs to make clinical decisions
    """
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Medical triage rules
        self.red_flag_rules = self._load_red_flag_rules()
        
        self.triage_system_prompt = """You are an expert emergency medicine physician and triage specialist. Your role is to:

1. Analyze patient presentation from multiple data sources
2. Identify red flag symptoms requiring immediate attention
3. Apply evidence-based triage criteria (ESI, CTAS, etc.)
4. Cross-validate information for contradictions
5. Make safe, appropriate disposition decisions
6. Provide clear clinical reasoning

Critical Safety Rules:
- When in doubt, triage UP (higher acuity)
- Always consider worst-case scenarios first
- Red flags ALWAYS trigger high urgency
- Document all clinical reasoning
- Confidence score < 0.7 → escalate to human review

Red Flag Symptoms (Always HIGH urgency):
- Chest pain with cardiac risk factors
- Sudden severe headache ("worst headache of life")
- Difficulty breathing at rest
- Altered mental status
- Signs of stroke (FAST criteria)
- Severe abdominal pain with peritoneal signs
- High fever with neck stiffness
- Uncontrolled bleeding
- Suicidal ideation

Special Populations:
- Elderly: Lower threshold for admission
- Immunocompromised: Higher vigilance
- Pediatrics: Different vital sign norms
- Pregnant: Consider fetal wellbeing"""
    
    def make_triage_decision(
        self,
        voice_analysis: Dict,
        symptom_analysis: Dict,
        document_data: Dict
    ) -> TriageDecision:
        """
        Main triage decision pipeline
        
        Args:
            voice_analysis: Output from Agent 1 (voice intake)
            symptom_analysis: Output from Agent 2 (symptom reasoning)
            document_data: Output from Agent 3 (document processing)
            
        Returns:
            TriageDecision with complete assessment
        """
        
        # Step 1: Check for red flags
        red_flags = self._identify_red_flags(symptom_analysis)
        
        # Step 2: Cross-validate data for contradictions
        contradictions = self._check_contradictions(symptom_analysis, document_data)
        
        # Step 3: Use Gemini for comprehensive clinical reasoning
        triage_analysis = self._perform_clinical_reasoning(
            voice_analysis,
            symptom_analysis,
            document_data,
            red_flags,
            contradictions
        )
        
        # Step 4: Determine final urgency and disposition
        urgency_level, confidence = self._calculate_urgency(
            triage_analysis,
            red_flags,
            voice_analysis
        )
        
        disposition = self._determine_disposition(urgency_level, symptom_analysis)
        
        # Step 5: Generate recommendations
        tests = self._recommend_tests(symptom_analysis)
        specialist = self._suggest_specialist(symptom_analysis)
        wait_time = self._estimate_wait_time(urgency_level)
        
        return TriageDecision(
            urgency_level=urgency_level,
            urgency_confidence=confidence,
            disposition=disposition,
            red_flags=red_flags,
            contradictions=contradictions,
            clinical_reasoning=triage_analysis.get("reasoning", ""),
            recommended_tests=tests,
            specialist_referral=specialist,
            estimated_wait_time=wait_time
        )
    
    def _load_red_flag_rules(self) -> Dict:
        """Load red flag symptom patterns"""
        return {
            "chest_pain": {
                "keywords": ["chest pain", "chest pressure", "chest tightness"],
                "risk_factors": ["hypertension", "diabetes", "smoking", "age>50"],
                "severity": "critical"
            },
            "stroke_symptoms": {
                "keywords": ["facial droop", "arm weakness", "speech difficulty", "sudden confusion"],
                "severity": "critical"
            },
            "severe_headache": {
                "keywords": ["worst headache", "thunderclap headache", "sudden severe headache"],
                "severity": "critical"
            },
            "respiratory_distress": {
                "keywords": ["can't breathe", "gasping", "shortness of breath at rest"],
                "severity": "critical"
            },
            "severe_bleeding": {
                "keywords": ["uncontrolled bleeding", "spurting blood"],
                "severity": "critical"
            },
            "altered_mental_status": {
                "keywords": ["confused", "disoriented", "unresponsive", "altered consciousness"],
                "severity": "critical"
            }
        }
    
    def _identify_red_flags(self, symptom_analysis: Dict) -> List[RedFlag]:
        """Identify red flag symptoms from analysis"""
        
        red_flags = []
        symptoms = symptom_analysis.get("symptoms", [])
        medical_history = symptom_analysis.get("medical_history", [])
        
        # Check each symptom against red flag rules
        for symptom in symptoms:
            symptom_name = symptom.get("name", "").lower()
            
            # Chest pain with risk factors
            if any(kw in symptom_name for kw in ["chest pain", "chest pressure"]):
                has_risk_factors = any(
                    risk in " ".join(medical_history).lower()
                    for risk in ["hypertension", "diabetes", "heart", "cardiac"]
                )
                
                if has_risk_factors or symptom.get("severity") in ["severe", "critical"]:
                    red_flags.append(RedFlag(
                        symptom="Chest Pain with Risk Factors",
                        reasoning="Chest pain in patient with cardiac risk factors requires immediate cardiac workup",
                        severity="critical"
                    ))
            
            # Severe headache
            if any(kw in symptom_name for kw in ["worst headache", "severe headache", "thunderclap"]):
                red_flags.append(RedFlag(
                    symptom="Severe Headache",
                    reasoning="Sudden severe headache raises concern for SAH, meningitis, or other neurological emergency",
                    severity="critical"
                ))
            
            # Respiratory distress
            if any(kw in symptom_name for kw in ["shortness of breath", "difficulty breathing", "can't breathe"]):
                if symptom.get("severity") in ["severe", "critical"]:
                    red_flags.append(RedFlag(
                        symptom="Respiratory Distress",
                        reasoning="Severe respiratory distress requires immediate evaluation and oxygen support",
                        severity="critical"
                    ))
            
            # Abdominal pain with peritoneal signs
            if "abdominal pain" in symptom_name:
                if symptom.get("severity") == "severe":
                    red_flags.append(RedFlag(
                        symptom="Severe Abdominal Pain",
                        reasoning="Severe abdominal pain may indicate surgical emergency (appendicitis, perforation, etc.)",
                        severity="high"
                    ))
        
        return red_flags
    
    def _check_contradictions(self, symptom_analysis: Dict, document_data: Dict) -> List[ClinicalContradiction]:
        """Cross-validate symptom data with document data"""
        
        contradictions = []
        
        # Check allergy contradictions
        stated_allergies = set(
            a.lower() for a in symptom_analysis.get("allergies", [])
        )
        
        stated_medications = set(
            m.lower() for m in symptom_analysis.get("current_medications", [])
        )
        
        # Check for penicillin allergy with amoxicillin use
        if "penicillin" in stated_allergies:
            dangerous_meds = ["amoxicillin", "ampicillin", "penicillin"]
            for med in stated_medications:
                if any(dm in med for dm in dangerous_meds):
                    contradictions.append(ClinicalContradiction(
                        finding="Penicillin allergy documented",
                        conflict=f"Patient currently taking {med} (penicillin-based)",
                        recommendation="STOP medication immediately, consult physician"
                    ))
        
        # Check medical record vs stated history
        if document_data.get("medical_record"):
            record = document_data["medical_record"]
            record_conditions = set(
                c.lower() for c in record.get("medical_conditions", [])
            )
            stated_conditions = set(
                c.lower() for c in symptom_analysis.get("medical_history", [])
            )
            
            # Find conditions in record but not mentioned
            missing_conditions = record_conditions - stated_conditions
            if missing_conditions and "asthma" in missing_conditions:
                contradictions.append(ClinicalContradiction(
                    finding="Asthma documented in medical record",
                    conflict="Patient did not mention asthma history",
                    recommendation="Clarify asthma history, consider pulmonary involvement"
                ))
        
        return contradictions
    
    def _perform_clinical_reasoning(
        self,
        voice_analysis: Dict,
        symptom_analysis: Dict,
        document_data: Dict,
        red_flags: List[RedFlag],
        contradictions: List[ClinicalContradiction]
    ) -> Dict:
        """Use Gemini for comprehensive clinical reasoning"""
        
        prompt = f"""{self.triage_system_prompt}

PATIENT PRESENTATION:

Voice Analysis:
- Urgency Level: {voice_analysis.get('urgency_level')}
- Emotional State: {voice_analysis.get('emotions')}

Symptoms:
{self._format_symptoms(symptom_analysis.get('symptoms', []))}

Medical History: {symptom_analysis.get('medical_history', [])}
Current Medications: {symptom_analysis.get('current_medications', [])}
Allergies: {symptom_analysis.get('allergies', [])}

Red Flags Identified: {len(red_flags)}
{self._format_red_flags(red_flags)}

Contradictions Found: {len(contradictions)}
{self._format_contradictions(contradictions)}

Document Data: {self._format_document_data(document_data)}

Please provide comprehensive triage analysis in JSON format:
{{
    "urgency_assessment": "immediate/urgent/semi_urgent/non_urgent",
    "confidence": 0.95,
    "clinical_reasoning": "Detailed explanation of triage decision",
    "differential_diagnosis": ["most likely conditions"],
    "recommended_disposition": "where patient should go",
    "recommended_tests": ["tests to order"],
    "safety_concerns": ["specific concerns"],
    "specialist_needed": "specialty if applicable"
}}"""
        
        response = self.model.generate_content(prompt)
        
        # Parse response
        import json
        import re
        
        json_match = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.text
        
        try:
            analysis = json.loads(json_str)
        except:
            analysis = {
                "urgency_assessment": "urgent",
                "confidence": 0.7,
                "clinical_reasoning": "Unable to parse detailed analysis",
                "recommended_tests": [],
                "safety_concerns": []
            }
        
        return analysis
    
    def _calculate_urgency(
        self,
        triage_analysis: Dict,
        red_flags: List[RedFlag],
        voice_analysis: Dict
    ) -> tuple[UrgencyLevel, float]:
        """Determine final urgency level with confidence score"""
        
        # Red flags always trigger high urgency
        if red_flags:
            if any(rf.severity == "critical" for rf in red_flags):
                return UrgencyLevel.IMMEDIATE, 0.95
            else:
                return UrgencyLevel.URGENT, 0.90
        
        # Use Gemini's assessment
        assessment = triage_analysis.get("urgency_assessment", "semi_urgent")
        confidence = triage_analysis.get("confidence", 0.75)
        
        # Factor in voice urgency
        voice_urgency = voice_analysis.get("urgency_level", "low")
        if voice_urgency == "high" and assessment != "immediate":
            # Voice suggests more urgency - adjust
            confidence *= 0.9
        
        urgency_map = {
            "immediate": UrgencyLevel.IMMEDIATE,
            "urgent": UrgencyLevel.URGENT,
            "semi_urgent": UrgencyLevel.SEMI_URGENT,
            "non_urgent": UrgencyLevel.NON_URGENT
        }
        
        urgency_level = urgency_map.get(assessment, UrgencyLevel.URGENT)
        
        return urgency_level, confidence
    
    def _determine_disposition(self, urgency: UrgencyLevel, symptom_analysis: Dict) -> DispositionType:
        """Determine where patient should go"""
        
        if urgency == UrgencyLevel.IMMEDIATE:
            return DispositionType.EMERGENCY_DEPARTMENT
        elif urgency == UrgencyLevel.URGENT:
            return DispositionType.URGENT_CARE
        elif urgency == UrgencyLevel.SEMI_URGENT:
            return DispositionType.PRIMARY_CARE_SAME_DAY
        else:
            return DispositionType.PRIMARY_CARE_ROUTINE
    
    def _recommend_tests(self, symptom_analysis: Dict) -> List[str]:
        """Recommend appropriate tests"""
        tests = []
        symptoms = [s.get("name", "").lower() for s in symptom_analysis.get("symptoms", [])]
        
        if any("chest pain" in s for s in symptoms):
            tests.extend(["ECG", "Troponin", "Chest X-ray"])
        
        if any("abdominal pain" in s for s in symptoms):
            tests.extend(["CBC", "CMP", "Lipase", "Urinalysis"])
        
        if any("headache" in s for s in symptoms):
            tests.extend(["CT Head non-contrast", "CBC"])
        
        if any("shortness of breath" in s or "cough" in s for s in symptoms):
            tests.extend(["Chest X-ray", "Pulse oximetry", "ABG if severe"])
        
        return tests
    
    def _suggest_specialist(self, symptom_analysis: Dict) -> Optional[str]:
        """Suggest specialist referral if needed"""
        symptoms = [s.get("name", "").lower() for s in symptom_analysis.get("symptoms", [])]
        
        if any("chest pain" in s for s in symptoms):
            return "Cardiology"
        if any("abdominal pain" in s for s in symptoms):
            return "Gastroenterology or General Surgery"
        if any("headache" in s for s in symptoms):
            return "Neurology"
        
        return None
    
    def _estimate_wait_time(self, urgency: UrgencyLevel) -> str:
        """Estimate wait time based on urgency"""
        wait_times = {
            UrgencyLevel.IMMEDIATE: "0 minutes - immediate attention",
            UrgencyLevel.URGENT: "15-30 minutes",
            UrgencyLevel.SEMI_URGENT: "1-2 hours",
            UrgencyLevel.NON_URGENT: "2-4 hours"
        }
        return wait_times.get(urgency, "Unknown")
    
    def _format_symptoms(self, symptoms: List[Dict]) -> str:
        """Format symptoms for prompt"""
        return "\n".join([
            f"- {s.get('name')}: {s.get('severity')} for {s.get('duration')}"
            for s in symptoms
        ])
    
    def _format_red_flags(self, red_flags: List[RedFlag]) -> str:
        """Format red flags for prompt"""
        if not red_flags:
            return "None"
        return "\n".join([
            f"- {rf.symptom}: {rf.reasoning}"
            for rf in red_flags
        ])
    
    def _format_contradictions(self, contradictions: List[ClinicalContradiction]) -> str:
        """Format contradictions for prompt"""
        if not contradictions:
            return "None"
        return "\n".join([
            f"- {c.finding} vs {c.conflict}"
            for c in contradictions
        ])
    
    def _format_document_data(self, document_data: Dict) -> str:
        """Format document data for prompt"""
        if not document_data:
            return "No documents processed"
        
        result = []
        if document_data.get("insurance_info"):
            ins = document_data["insurance_info"]
            result.append(f"Insurance: {ins.get('provider')} - {ins.get('member_id')}")
        
        if document_data.get("medical_record"):
            rec = document_data["medical_record"]
            result.append(f"Medical Record: {rec.get('medical_conditions')}")
        
        return "\n".join(result) if result else "No documents processed"


# Demo usage
if __name__ == "__main__":
    agent = TriageCoordinatorAgent()
    
    # Sample data from previous agents
    voice_analysis = {
        "urgency_level": "high",
        "emotions": {"stress": 0.7, "pain": 0.6}
    }
    
    symptom_analysis = {
        "symptoms": [
            {"name": "chest pain", "severity": "severe", "duration": "2 days"},
            {"name": "fever", "severity": "moderate", "duration": "2 days"}
        ],
        "medical_history": ["hypertension", "asthma"],
        "current_medications": ["lisinopril"],
        "allergies": ["penicillin"]
    }
    
    document_data = {
        "insurance_info": {"provider": "Blue Cross", "member_id": "ABC123"},
        "medical_record": {"medical_conditions": ["hypertension", "asthma"]}
    }
    
    decision = agent.make_triage_decision(voice_analysis, symptom_analysis, document_data)
    
    print(f"Urgency: {decision.urgency_level.value} (confidence: {decision.urgency_confidence:.2%})")
    print(f"Disposition: {decision.disposition.value}")
    print(f"Red Flags: {len(decision.red_flags)}")
    print(f"Recommended Tests: {decision.recommended_tests}")