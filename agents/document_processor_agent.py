"""
Agent 3: Document Processing Specialist
Uses Google Vision API to extract data from medical documents
"""

import os
from google.cloud import vision
from google.cloud import documentai_v1 as documentai
from typing import Dict, List, Optional
from dataclasses import dataclass
from PIL import Image
import io


@dataclass
class InsuranceInfo:
    provider: str
    member_id: str
    group_number: Optional[str]
    plan_type: str
    coverage_status: str


@dataclass
class MedicalRecordInfo:
    patient_name: str
    date_of_birth: Optional[str]
    medical_conditions: List[str]
    medications: List[str]
    allergies: List[str]
    last_visit_date: Optional[str]
    provider_name: Optional[str]


@dataclass
class PrescriptionInfo:
    medication_name: str
    dosage: str
    frequency: str
    prescriber: str
    date_prescribed: Optional[str]


@dataclass
class DocumentProcessingResult:
    document_type: str  # "insurance_card", "medical_record", "prescription"
    insurance_info: Optional[InsuranceInfo]
    medical_record: Optional[MedicalRecordInfo]
    prescription: Optional[PrescriptionInfo]
    extracted_text: str
    confidence_score: float
    validation_warnings: List[str]


class DocumentProcessorAgent:
    """
    Document Processing Specialist using Google Vision API
    Extracts structured data from medical documents
    """
    
    def __init__(self):
        self.vision_client = vision.ImageAnnotatorClient()
        
        # For demo: we'll focus on Vision API text detection
        # In production, you'd also use Document AI for structured extraction
        
        self.document_patterns = {
            "insurance_card": {
                "keywords": ["insurance", "member", "id", "group", "coverage", "plan"],
                "required_fields": ["member_id", "provider"]
            },
            "medical_record": {
                "keywords": ["patient", "diagnosis", "history", "medications", "allergies"],
                "required_fields": ["patient_name"]
            },
            "prescription": {
                "keywords": ["rx", "prescription", "sig", "refill", "dosage"],
                "required_fields": ["medication_name", "dosage"]
            }
        }
    
    def process_document(self, image_path: str) -> DocumentProcessingResult:
        """
        Main document processing pipeline
        
        Args:
            image_path: Path to document image
            
        Returns:
            DocumentProcessingResult with extracted data
        """
        
        # Step 1: Detect document type
        extracted_text = self._extract_text_from_image(image_path)
        document_type = self._classify_document(extracted_text)
        
        # Step 2: Extract structured data based on type
        if document_type == "insurance_card":
            structured_data = self._extract_insurance_info(extracted_text)
            result = DocumentProcessingResult(
                document_type=document_type,
                insurance_info=structured_data["data"],
                medical_record=None,
                prescription=None,
                extracted_text=extracted_text,
                confidence_score=structured_data["confidence"],
                validation_warnings=structured_data["warnings"]
            )
        
        elif document_type == "medical_record":
            structured_data = self._extract_medical_record(extracted_text)
            result = DocumentProcessingResult(
                document_type=document_type,
                insurance_info=None,
                medical_record=structured_data["data"],
                prescription=None,
                extracted_text=extracted_text,
                confidence_score=structured_data["confidence"],
                validation_warnings=structured_data["warnings"]
            )
        
        elif document_type == "prescription":
            structured_data = self._extract_prescription_info(extracted_text)
            result = DocumentProcessingResult(
                document_type=document_type,
                insurance_info=None,
                medical_record=None,
                prescription=structured_data["data"],
                extracted_text=extracted_text,
                confidence_score=structured_data["confidence"],
                validation_warnings=structured_data["warnings"]
            )
        
        else:
            result = DocumentProcessingResult(
                document_type="unknown",
                insurance_info=None,
                medical_record=None,
                prescription=None,
                extracted_text=extracted_text,
                confidence_score=0.3,
                validation_warnings=["Could not determine document type"]
            )
        
        return result
    
    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract all text from image using Vision API"""
        
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        
        # Use document text detection for better accuracy
        response = self.vision_client.document_text_detection(image=image)
        
        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")
        
        # Get full text
        text = response.full_text_annotation.text
        
        return text
    
    def _classify_document(self, text: str) -> str:
        """Classify document type based on keywords"""
        
        text_lower = text.lower()
        scores = {}
        
        for doc_type, patterns in self.document_patterns.items():
            keyword_matches = sum(
                1 for keyword in patterns["keywords"] 
                if keyword in text_lower
            )
            scores[doc_type] = keyword_matches
        
        # Return type with highest score
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return "unknown"
    
    def _extract_insurance_info(self, text: str) -> Dict:
        """Extract insurance card information"""
        import re
        
        warnings = []
        
        # Extract member ID (various patterns)
        member_id = None
        id_patterns = [
            r'member\s*id[:\s]+([A-Z0-9]+)',
            r'id[:\s]+([A-Z0-9]{8,})',
            r'member\s*#[:\s]+([A-Z0-9]+)'
        ]
        
        for pattern in id_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                member_id = match.group(1)
                break
        
        if not member_id:
            warnings.append("Member ID not found")
        
        # Extract group number
        group_match = re.search(r'group[:\s#]+([A-Z0-9]+)', text, re.IGNORECASE)
        group_number = group_match.group(1) if group_match else None
        
        # Identify provider (common insurance companies)
        providers = ["blue cross", "aetna", "cigna", "united", "humana", "kaiser"]
        provider = None
        for p in providers:
            if p in text.lower():
                provider = p.title()
                break
        
        if not provider:
            provider = "Unknown Provider"
            warnings.append("Insurance provider not identified")
        
        # Determine plan type
        plan_type = "Unknown"
        if "ppo" in text.lower():
            plan_type = "PPO"
        elif "hmo" in text.lower():
            plan_type = "HMO"
        elif "epo" in text.lower():
            plan_type = "EPO"
        
        insurance_info = InsuranceInfo(
            provider=provider,
            member_id=member_id or "NOT_FOUND",
            group_number=group_number,
            plan_type=plan_type,
            coverage_status="Active"  # Assume active for demo
        )
        
        confidence = 0.9 if member_id and provider != "Unknown Provider" else 0.6
        
        return {
            "data": insurance_info,
            "confidence": confidence,
            "warnings": warnings
        }
    
    def _extract_medical_record(self, text: str) -> Dict:
        """Extract medical record information"""
        import re
        
        warnings = []
        
        # Extract patient name (usually first line or after "Patient:")
        name_match = re.search(r'patient[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
        patient_name = name_match.group(1).strip() if name_match else "Unknown Patient"
        
        # Extract conditions
        conditions = []
        condition_keywords = ["diagnosis", "history", "conditions"]
        for keyword in condition_keywords:
            pattern = f"{keyword}[:\s]+([^\\n]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                conditions.extend([c.strip() for c in match.group(1).split(",")])
        
        # Extract medications
        medications = []
        med_match = re.search(r'medications?[:\s]+([^\\n]+)', text, re.IGNORECASE)
        if med_match:
            medications = [m.strip() for m in med_match.group(1).split(",")]
        
        # Extract allergies
        allergies = []
        allergy_match = re.search(r'allergies[:\s]+([^\\n]+)', text, re.IGNORECASE)
        if allergy_match:
            allergy_text = allergy_match.group(1)
            if "nka" in allergy_text.lower() or "none" in allergy_text.lower():
                allergies = ["No known allergies"]
            else:
                allergies = [a.strip() for a in allergy_text.split(",")]
        
        medical_record = MedicalRecordInfo(
            patient_name=patient_name,
            date_of_birth=None,  # Would need more sophisticated extraction
            medical_conditions=conditions,
            medications=medications,
            allergies=allergies,
            last_visit_date=None,
            provider_name=None
        )
        
        if not conditions and not medications:
            warnings.append("Limited medical information extracted")
        
        confidence = 0.8 if patient_name != "Unknown Patient" else 0.5
        
        return {
            "data": medical_record,
            "confidence": confidence,
            "warnings": warnings
        }
    
    def _extract_prescription_info(self, text: str) -> Dict:
        """Extract prescription information"""
        import re
        
        warnings = []
        
        # Extract medication name (usually prominent)
        lines = text.split('\n')
        medication_name = lines[0] if lines else "Unknown Medication"
        
        # Extract dosage
        dosage_match = re.search(r'(\d+\s*mg|\d+\s*mcg)', text, re.IGNORECASE)
        dosage = dosage_match.group(1) if dosage_match else "Unknown dosage"
        
        # Extract frequency
        frequency = "Unknown frequency"
        freq_patterns = [
            r'(once|twice|three times)\s*(daily|per day)',
            r'(\d+)\s*times?\s*(daily|per day)',
            r'every\s*(\d+)\s*hours'
        ]
        for pattern in freq_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                frequency = match.group(0)
                break
        
        # Extract prescriber
        prescriber_match = re.search(r'dr\.?\s+([A-Za-z\s]+)', text, re.IGNORECASE)
        prescriber = prescriber_match.group(1).strip() if prescriber_match else "Unknown Doctor"
        
        prescription = PrescriptionInfo(
            medication_name=medication_name,
            dosage=dosage,
            frequency=frequency,
            prescriber=prescriber,
            date_prescribed=None
        )
        
        if dosage == "Unknown dosage":
            warnings.append("Dosage information incomplete")
        
        confidence = 0.75
        
        return {
            "data": prescription,
            "confidence": confidence,
            "warnings": warnings
        }


# Demo usage
if __name__ == "__main__":
    agent = DocumentProcessorAgent()
    
    # Process insurance card
    result = agent.process_document("sample_insurance_card.jpg")
    
    print(f"Document Type: {result.document_type}")
    print(f"Confidence: {result.confidence_score:.2%}")
    
    if result.insurance_info:
        print(f"\nInsurance Info:")
        print(f"  Provider: {result.insurance_info.provider}")
        print(f"  Member ID: {result.insurance_info.member_id}")
        print(f"  Plan Type: {result.insurance_info.plan_type}")
    
    if result.validation_warnings:
        print(f"\nWarnings: {result.validation_warnings}")