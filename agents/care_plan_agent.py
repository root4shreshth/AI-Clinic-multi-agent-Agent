"""
Agent 5: Care Plan Generator
Generates personalized care plans and instructions for different audiences
"""

import os
import google.generativeai as genai
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class DoctorSummary:
    patient_demographics: str
    chief_complaint: str
    history_of_present_illness: str
    past_medical_history: List[str]
    medications: List[str]
    allergies: List[str]
    physical_exam_notes: str
    triage_assessment: str
    red_flags: List[str]
    differential_diagnosis: List[str]
    recommended_workup: List[str]
    disposition_plan: str


@dataclass
class PatientInstructions:
    welcome_message: str
    what_to_expect: List[str]
    estimated_wait_time: str
    preparation_steps: List[str]
    warning_signs: List[str]
    follow_up_instructions: str
    language: str


@dataclass
class SystemActions:
    appointments_to_schedule: List[Dict]
    alerts_to_send: List[Dict]
    tests_to_order: List[str]
    referrals_to_create: List[Dict]
    notifications: List[Dict]


@dataclass
class CarePlan:
    doctor_summary: DoctorSummary
    patient_instructions: PatientInstructions
    system_actions: SystemActions
    generated_at: datetime


class CarePlanAgent:
    """
    Care Plan Generator - Creates personalized outputs for all stakeholders
    """
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def generate_care_plan(
        self,
        patient_data: Dict,
        triage_decision: Dict,
        all_agent_data: Dict
    ) -> CarePlan:
        """
        Generate comprehensive care plan for all stakeholders
        
        Args:
            patient_data: Combined patient information
            triage_decision: Output from triage coordinator
            all_agent_data: All data from previous agents
            
        Returns:
            Complete CarePlan with all outputs
        """
        
        # Generate doctor summary
        doctor_summary = self._generate_doctor_summary(
            patient_data,
            triage_decision,
            all_agent_data
        )
        
        # Generate patient instructions
        patient_instructions = self._generate_patient_instructions(
            patient_data,
            triage_decision
        )
        
        # Generate system actions
        system_actions = self._generate_system_actions(
            triage_decision,
            patient_data
        )
        
        return CarePlan(
            doctor_summary=doctor_summary,
            patient_instructions=patient_instructions,
            system_actions=system_actions,
            generated_at=datetime.now()
        )
    
    def _generate_doctor_summary(
        self,
        patient_data: Dict,
        triage_decision: Dict,
        all_agent_data: Dict
    ) -> DoctorSummary:
        """Generate structured clinical summary for physicians"""
        
        prompt = f"""Generate a concise, professional clinical summary for a physician in standard SOAP format.

PATIENT DATA:
{self._format_patient_data(patient_data)}

TRIAGE ASSESSMENT:
- Urgency: {triage_decision.get('urgency_level')}
- Disposition: {triage_decision.get('disposition')}
- Red Flags: {triage_decision.get('red_flags', [])}
- Clinical Reasoning: {triage_decision.get('clinical_reasoning')}

Generate a professional summary with:
1. Patient demographics (age, gender if known)
2. Chief complaint
3. History of present illness (HPI)
4. Past medical history (PMH)
5. Medications
6. Allergies
7. Physical exam notes (from voice analysis if applicable)
8. Assessment with red flags highlighted
9. Differential diagnosis
10. Recommended workup
11. Disposition plan

Use standard medical abbreviations. Be concise but complete. Highlight RED FLAGS in caps."""
        
        response = self.model.generate_content(prompt)
        summary_text = response.text
        
        # Parse structured components
        symptoms = patient_data.get("symptoms", [])
        chief_complaint = symptoms[0].get("name") if symptoms else "Unknown"
        
        return DoctorSummary(
            patient_demographics=self._extract_demographics(patient_data),
            chief_complaint=chief_complaint,
            history_of_present_illness=self._generate_hpi(patient_data),
            past_medical_history=patient_data.get("medical_history", []),
            medications=patient_data.get("medications", []),
            allergies=patient_data.get("allergies", []),
            physical_exam_notes=self._generate_physical_exam_notes(all_agent_data),
            triage_assessment=f"{triage_decision.get('urgency_level')} - {triage_decision.get('clinical_reasoning', '')}",
            red_flags=[rf.get("symptom", "") for rf in triage_decision.get("red_flags", [])],
            differential_diagnosis=self._extract_differential(summary_text),
            recommended_workup=triage_decision.get("recommended_tests", []),
            disposition_plan=triage_decision.get("disposition", "")
        )
    
    def _generate_patient_instructions(
        self,
        patient_data: Dict,
        triage_decision: Dict
    ) -> PatientInstructions:
        """Generate patient-friendly care instructions"""
        
        urgency = triage_decision.get("urgency_level", "non_urgent")
        disposition = triage_decision.get("disposition", "")
        
        prompt = f"""Generate clear, reassuring, patient-friendly instructions for someone with:

Symptoms: {[s.get('name') for s in patient_data.get('symptoms', [])]}
Urgency: {urgency}
Where to go: {disposition}

Create:
1. A warm welcome message that acknowledges their concerns
2. What to expect during their visit (3-5 steps)
3. Estimated wait time: {triage_decision.get('estimated_wait_time')}
4. What they should do to prepare (bring items, fasting, etc.)
5. Warning signs that mean they should seek immediate help
6. Follow-up instructions

Use simple, non-medical language. Be empathetic and reassuring. Address common concerns."""
        
        response = self.model.generate_content(prompt)
        instructions_text = response.text
        
        # Parse components
        welcome_message = self._extract_welcome_message(instructions_text)
        what_to_expect = self._extract_list_items(instructions_text, "What to expect")
        preparation_steps = self._extract_list_items(instructions_text, "prepare")
        warning_signs = self._extract_warning_signs(urgency, patient_data)
        
        return PatientInstructions(
            welcome_message=welcome_message,
            what_to_expect=what_to_expect,
            estimated_wait_time=triage_decision.get("estimated_wait_time", "Unknown"),
            preparation_steps=preparation_steps,
            warning_signs=warning_signs,
            follow_up_instructions=self._generate_follow_up(urgency),
            language="en"
        )
    
    def _generate_system_actions(
        self,
        triage_decision: Dict,
        patient_data: Dict
    ) -> SystemActions:
        """Generate automated system actions"""
        
        urgency = triage_decision.get("urgency_level")
        disposition = triage_decision.get("disposition")
        
        # Schedule appointments
        appointments = []
        if urgency in ["immediate", "urgent"]:
            appointments.append({
                "type": "immediate_visit",
                "location": disposition,
                "time": "ASAP",
                "priority": "high"
            })
        
        # Create alerts
        alerts = []
        if triage_decision.get("red_flags"):
            alerts.append({
                "type": "clinical_alert",
                "severity": "high",
                "message": f"RED FLAG: {len(triage_decision['red_flags'])} critical findings",
                "recipients": ["triage_nurse", "attending_physician"]
            })
        
        # Order tests
        tests_to_order = triage_decision.get("recommended_tests", [])
        
        # Create referrals
        referrals = []
        if triage_decision.get("specialist_referral"):
            referrals.append({
                "specialty": triage_decision["specialist_referral"],
                "urgency": urgency,
                "reason": "Based on presenting symptoms"
            })
        
        # Notifications
        notifications = []
        notifications.append({
            "type": "patient_sms",
            "message": f"Your check-in is complete. Estimated wait: {triage_decision.get('estimated_wait_time')}",
            "send_at": "immediate"
        })
        
        if urgency == "immediate":
            notifications.append({
                "type": "staff_alert",
                "message": "HIGH PRIORITY patient in waiting room",
                "recipients": ["charge_nurse", "ER_physician"]
            })
        
        return SystemActions(
            appointments_to_schedule=appointments,
            alerts_to_send=alerts,
            tests_to_order=tests_to_order,
            referrals_to_create=referrals,
            notifications=notifications
        )
    
    # Helper methods
    def _format_patient_data(self, patient_data: Dict) -> str:
        """Format patient data for prompts"""
        symptoms = "\n".join([
            f"- {s.get('name')}: {s.get('severity')} for {s.get('duration')}"
            for s in patient_data.get("symptoms", [])
        ])
        
        return f"""
Symptoms:
{symptoms}

Medical History: {patient_data.get('medical_history', [])}
Medications: {patient_data.get('medications', [])}
Allergies: {patient_data.get('allergies', [])}
"""
    
    def _extract_demographics(self, patient_data: Dict) -> str:
        """Extract or infer patient demographics"""
        # In real system, this would come from registration
        # For demo, we infer from medical history
        
        history = patient_data.get("medical_history", [])
        
        age_group = "adult"
        if "pediatric" in " ".join(history).lower():
            age_group = "pediatric"
        elif any(term in " ".join(history).lower() for term in ["elderly", "geriatric"]):
            age_group = "elderly"
        
        return f"{age_group}, gender unknown"
    
    def _generate_hpi(self, patient_data: Dict) -> str:
        """Generate History of Present Illness"""
        symptoms = patient_data.get("symptoms", [])
        
        if not symptoms:
            return "Patient presents with undifferentiated symptoms"
        
        primary_symptom = symptoms[0]
        hpi_parts = []
        
        hpi_parts.append(f"Patient reports {primary_symptom.get('name')}")
        
        if primary_symptom.get("duration"):
            hpi_parts.append(f"for {primary_symptom.get('duration')}")
        
        if primary_symptom.get("severity"):
            hpi_parts.append(f"described as {primary_symptom.get('severity')}")
        
        if len(symptoms) > 1:
            associated = ", ".join([s.get('name') for s in symptoms[1:]])
            hpi_parts.append(f"Associated symptoms include {associated}")
        
        return ". ".join(hpi_parts) + "."
    
    def _generate_physical_exam_notes(self, all_agent_data: Dict) -> str:
        """Generate PE notes from voice analysis"""
        voice_data = all_agent_data.get("voice_analysis", {})
        
        notes = []
        
        # Infer from voice urgency
        urgency = voice_data.get("urgency_level", "")
        emotions = voice_data.get("emotions", {})
        
        if urgency == "high":
            notes.append("Patient appears in distress")
        
        if emotions.get("pain", 0) > 0.6:
            notes.append("Signs of pain evident in voice")
        
        if emotions.get("anxiety", 0) > 0.6:
            notes.append("Patient appears anxious")
        
        return "; ".join(notes) if notes else "Unable to assess remotely"
    
    def _extract_differential(self, summary_text: str) -> List[str]:
        """Extract differential diagnosis from summary"""
        # Simple extraction - in production would be more sophisticated
        differentials = []
        
        # Look for common patterns
        if "cardiac" in summary_text.lower() or "mi" in summary_text.lower():
            differentials.append("Acute coronary syndrome")
        
        if "pneumonia" in summary_text.lower():
            differentials.append("Pneumonia")
        
        if "appendicitis" in summary_text.lower():
            differentials.append("Appendicitis")
        
        return differentials if differentials else ["Diagnosis pending further evaluation"]
    
    def _extract_welcome_message(self, text: str) -> str:
        """Extract welcome message from generated text"""
        lines = text.split('\n')
        for line in lines[:3]:  # First few lines usually have welcome
            if len(line) > 20 and not line.startswith('#'):
                return line.strip()
        
        return "Thank you for checking in. We're here to help you."
    
    def _extract_list_items(self, text: str, section_keyword: str) -> List[str]:
        """Extract list items from a section"""
        items = []
        in_section = False
        
        for line in text.split('\n'):
            if section_keyword.lower() in line.lower():
                in_section = True
                continue
            
            if in_section:
                if line.strip().startswith('-') or line.strip().startswith('•'):
                    items.append(line.strip()[1:].strip())
                elif line.strip() and not items:
                    continue
                elif items:
                    break
        
        return items[:5]  # Limit to 5 items
    
    def _extract_warning_signs(self, urgency: str, patient_data: Dict) -> List[str]:
        """Generate warning signs specific to patient's condition"""
        warning_signs = [
            "Severe or worsening pain",
            "Difficulty breathing or shortness of breath",
            "Chest pain or pressure",
            "High fever over 103°F",
            "Severe bleeding",
            "Confusion or difficulty staying awake",
            "Severe allergic reaction (hives, swelling, difficulty breathing)"
        ]
        
        # Customize based on symptoms
        symptoms = [s.get("name", "").lower() for s in patient_data.get("symptoms", [])]
        
        if any("chest" in s for s in symptoms):
            warning_signs.insert(0, "Pain spreading to arm, jaw, or back")
        
        if any("abdominal" in s for s in symptoms):
            warning_signs.insert(0, "Severe abdominal pain or rigid belly")
        
        if any("headache" in s for s in symptoms):
            warning_signs.insert(0, "Worst headache of your life or sudden severe headache")
        
        return warning_signs[:7]  # Return top 7
    
    def _generate_follow_up(self, urgency: str) -> str:
        """Generate follow-up instructions"""
        if urgency in ["immediate", "urgent"]:
            return "Follow up with your doctor within 24-48 hours or as instructed by the treating physician."
        elif urgency == "semi_urgent":
            return "Schedule a follow-up appointment with your primary care doctor within 3-7 days."
        else:
            return "Follow up with your primary care doctor as needed or within 1-2 weeks."


# Demo usage
if __name__ == "__main__":
    agent = CarePlanAgent()
    
    patient_data = {
        "symptoms": [
            {"name": "chest pain", "severity": "severe", "duration": "2 days"}
        ],
        "medical_history": ["hypertension"],
        "medications": ["lisinopril"],
        "allergies": ["penicillin"]
    }
    
    triage_decision = {
        "urgency_level": "urgent",
        "disposition": "emergency_department",
        "red_flags": [{"symptom": "chest pain with risk factors"}],
        "recommended_tests": ["ECG", "Troponin", "Chest X-ray"],
        "estimated_wait_time": "15-30 minutes",
        "clinical_reasoning": "Chest pain with cardiac risk factors"
    }
    
    care_plan = agent.generate_care_plan(patient_data, triage_decision, {})
    
    print("=== DOCTOR SUMMARY ===")
    print(f"Chief Complaint: {care_plan.doctor_summary.chief_complaint}")
    print(f"Red Flags: {care_plan.doctor_summary.red_flags}")
    print(f"Recommended Workup: {care_plan.doctor_summary.recommended_workup}")
    
    print("\n=== PATIENT INSTRUCTIONS ===")
    print(care_plan.patient_instructions.welcome_message)
    print(f"Wait Time: {care_plan.patient_instructions.estimated_wait_time}")
    
    print("\n=== SYSTEM ACTIONS ===")
    print(f"Alerts to Send: {len(care_plan.system_actions.alerts_to_send)}")
    print(f"Tests to Order: {care_plan.system_actions.tests_to_order}")