"""
Agent Orchestrator using Google's Agent Development Kit (ADK)
Coordinates all 5 agents in the MediScan system
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

# Import our custom agents
import sys
sys.path.append('..')
from agents.voice_intake_agent import VoiceIntakeAgent, VoiceIntakeResult
from agents.symptom_reasoning_agent import SymptomReasoningAgent, SymptomAnalysisResult
from agents.document_processor_agent import DocumentProcessorAgent, DocumentProcessingResult
from agents.triage_coordinator_agent import TriageCoordinatorAgent, TriageDecision
from agents.care_plan_agent import CarePlanAgent, CarePlan


@dataclass
class PatientCase:
    case_id: str
    audio_file_path: Optional[str]
    document_image_paths: List[str]
    created_at: datetime
    
    # Agent results
    voice_result: Optional[VoiceIntakeResult] = None
    symptom_result: Optional[SymptomAnalysisResult] = None
    document_results: List[DocumentProcessingResult] = None
    triage_decision: Optional[TriageDecision] = None
    care_plan: Optional[CarePlan] = None
    
    # Status tracking
    current_stage: str = "initialized"
    progress_percentage: int = 0
    errors: List[str] = None


class MediScanOrchestrator:
    """
    Multi-Agent Orchestrator for MediScan Voice Triage System
    
    Coordinates the workflow across all 5 agents:
    1. Voice Intake Specialist
    2. Symptom Reasoning Engine
    3. Document Processing Specialist
    4. Clinical Triage Coordinator
    5. Care Plan Generator
    """
    
    def __init__(self):
        # Initialize all agents
        self.voice_agent = VoiceIntakeAgent()
        self.symptom_agent = SymptomReasoningAgent()
        self.document_agent = DocumentProcessorAgent()
        self.triage_agent = TriageCoordinatorAgent()
        self.care_plan_agent = CarePlanAgent()
        
        # Workflow configuration
        self.workflow_stages = [
            "voice_intake",
            "symptom_reasoning",
            "document_processing",
            "triage_coordination",
            "care_plan_generation"
        ]
        
        # Active cases tracking
        self.active_cases: Dict[str, PatientCase] = {}
    
    async def process_patient(
        self,
        case_id: str,
        audio_file: Optional[str] = None,
        document_images: List[str] = None,
        progress_callback: Optional[callable] = None
    ) -> PatientCase:
        """
        Main orchestration method - processes a complete patient case
        
        Args:
            case_id: Unique case identifier
            audio_file: Path to patient voice audio
            document_images: List of paths to document images
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete PatientCase with all results
        """
        
        # Create case
        case = PatientCase(
            case_id=case_id,
            audio_file_path=audio_file,
            document_image_paths=document_images or [],
            created_at=datetime.now(),
            errors=[]
        )
        
        self.active_cases[case_id] = case
        
        try:
            # Stage 1: Voice Intake (if audio provided)
            if audio_file:
                await self._update_progress(case, "voice_intake", 0, progress_callback)
                case.voice_result = await self._run_voice_intake(audio_file)
                await self._update_progress(case, "voice_intake", 20, progress_callback)
            
            # Stage 2: Symptom Reasoning
            await self._update_progress(case, "symptom_reasoning", 20, progress_callback)
            
            if case.voice_result:
                urgency_context = {
                    "urgency_level": case.voice_result.urgency_level,
                    "emotions": case.voice_result.speaker_emotions,
                    "duration": case.voice_result.duration_seconds
                }
                case.symptom_result = await self._run_symptom_reasoning(
                    case.voice_result.transcript,
                    urgency_context
                )
            else:
                # Fallback for demo without audio
                case.symptom_result = await self._run_symptom_reasoning(
                    "Sample patient transcript",
                    {}
                )
            
            await self._update_progress(case, "symptom_reasoning", 40, progress_callback)
            
            # Stage 3: Document Processing (parallel)
            await self._update_progress(case, "document_processing", 40, progress_callback)
            
            if document_images:
                case.document_results = await self._run_document_processing(document_images)
            else:
                case.document_results = []
            
            await self._update_progress(case, "document_processing", 60, progress_callback)
            
            # Stage 4: Triage Coordination (The Brain)
            await self._update_progress(case, "triage_coordination", 60, progress_callback)
            
            case.triage_decision = await self._run_triage_coordination(case)
            
            await self._update_progress(case, "triage_coordination", 80, progress_callback)
            
            # Stage 5: Care Plan Generation
            await self._update_progress(case, "care_plan_generation", 80, progress_callback)
            
            case.care_plan = await self._run_care_plan_generation(case)
            
            await self._update_progress(case, "care_plan_generation", 100, progress_callback)
            
            case.current_stage = "completed"
            
        except Exception as e:
            case.errors.append(f"Processing error: {str(e)}")
            case.current_stage = "failed"
            raise
        
        return case
    
    async def _run_voice_intake(self, audio_file: str) -> VoiceIntakeResult:
        """Run Voice Intake Agent (Agent 1)"""
        print(f"[Agent 1] Processing voice input: {audio_file}")
        
        # Run synchronously in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.voice_agent.process_audio,
            audio_file
        )
        
        print(f"[Agent 1] Complete - Urgency: {result.urgency_level}")
        return result
    
    async def _run_symptom_reasoning(
        self,
        transcript: str,
        urgency_context: Dict
    ) -> SymptomAnalysisResult:
        """Run Symptom Reasoning Agent (Agent 2)"""
        print(f"[Agent 2] Analyzing symptoms from transcript...")
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.symptom_agent.analyze_symptoms,
            transcript,
            urgency_context
        )
        
        print(f"[Agent 2] Complete - Found {len(result.symptoms)} symptoms")
        return result
    
    async def _run_document_processing(
        self,
        document_images: List[str]
    ) -> List[DocumentProcessingResult]:
        """Run Document Processing Agent (Agent 3) - parallel processing"""
        print(f"[Agent 3] Processing {len(document_images)} documents...")
        
        # Process documents in parallel
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                None,
                self.document_agent.process_document,
                image_path
            )
            for image_path in document_images
        ]
        
        results = await asyncio.gather(*tasks)
        
        print(f"[Agent 3] Complete - Processed {len(results)} documents")
        return results
    
    async def _run_triage_coordination(self, case: PatientCase) -> TriageDecision:
        """Run Triage Coordinator Agent (Agent 4) - The Brain"""
        print(f"[Agent 4] Coordinating triage decision...")
        
        # Prepare data from all previous agents
        voice_analysis = {}
        if case.voice_result:
            voice_analysis = {
                "urgency_level": case.voice_result.urgency_level,
                "emotions": case.voice_result.speaker_emotions,
                "duration": case.voice_result.duration_seconds
            }
        
        symptom_analysis = {}
        if case.symptom_result:
            symptom_analysis = {
                "symptoms": [
                    {
                        "name": s.name,
                        "severity": s.severity.value,
                        "duration": s.duration,
                        "location": s.location,
                        "characteristics": s.characteristics
                    }
                    for s in case.symptom_result.symptoms
                ],
                "medical_history": case.symptom_result.medical_history,
                "current_medications": case.symptom_result.current_medications,
                "allergies": case.symptom_result.allergies
            }
        
        document_data = {}
        if case.document_results:
            for doc_result in case.document_results:
                if doc_result.insurance_info:
                    document_data["insurance_info"] = {
                        "provider": doc_result.insurance_info.provider,
                        "member_id": doc_result.insurance_info.member_id,
                        "plan_type": doc_result.insurance_info.plan_type
                    }
                if doc_result.medical_record:
                    document_data["medical_record"] = {
                        "patient_name": doc_result.medical_record.patient_name,
                        "medical_conditions": doc_result.medical_record.medical_conditions,
                        "medications": doc_result.medical_record.medications,
                        "allergies": doc_result.medical_record.allergies
                    }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.triage_agent.make_triage_decision,
            voice_analysis,
            symptom_analysis,
            document_data
        )
        
        print(f"[Agent 4] Complete - Urgency: {result.urgency_level.value}")
        return result
    
    async def _run_care_plan_generation(self, case: PatientCase) -> CarePlan:
        """Run Care Plan Generator (Agent 5)"""
        print(f"[Agent 5] Generating care plans...")
        
        # Consolidate all patient data
        patient_data = {}
        if case.symptom_result:
            patient_data = {
                "symptoms": [
                    {
                        "name": s.name,
                        "severity": s.severity.value,
                        "duration": s.duration
                    }
                    for s in case.symptom_result.symptoms
                ],
                "medical_history": case.symptom_result.medical_history,
                "medications": case.symptom_result.current_medications,
                "allergies": case.symptom_result.allergies
            }
        
        triage_data = {}
        if case.triage_decision:
            triage_data = {
                "urgency_level": case.triage_decision.urgency_level.value,
                "disposition": case.triage_decision.disposition.value,
                "red_flags": [
                    {
                        "symptom": rf.symptom,
                        "reasoning": rf.reasoning
                    }
                    for rf in case.triage_decision.red_flags
                ],
                "recommended_tests": case.triage_decision.recommended_tests,
                "specialist_referral": case.triage_decision.specialist_referral,
                "estimated_wait_time": case.triage_decision.estimated_wait_time,
                "clinical_reasoning": case.triage_decision.clinical_reasoning
            }
        
        all_agent_data = {
            "voice_analysis": {
                "urgency_level": case.voice_result.urgency_level if case.voice_result else "unknown",
                "emotions": case.voice_result.speaker_emotions if case.voice_result else {}
            }
        }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.care_plan_agent.generate_care_plan,
            patient_data,
            triage_data,
            all_agent_data
        )
        
        print(f"[Agent 5] Complete - Care plan generated")
        return result
    
    async def _update_progress(
        self,
        case: PatientCase,
        stage: str,
        percentage: int,
        callback: Optional[callable] = None
    ):
        """Update case progress"""
        case.current_stage = stage
        case.progress_percentage = percentage
        
        if callback:
            await callback(case.case_id, stage, percentage)
    
    def get_case_status(self, case_id: str) -> Optional[Dict]:
        """Get current status of a case"""
        case = self.active_cases.get(case_id)
        
        if not case:
            return None
        
        return {
            "case_id": case.case_id,
            "current_stage": case.current_stage,
            "progress_percentage": case.progress_percentage,
            "errors": case.errors
        }
    
    def export_case_summary(self, case_id: str) -> Dict:
        """Export complete case summary for demo/presentation"""
        case = self.active_cases.get(case_id)
        
        if not case:
            return {"error": "Case not found"}
        
        summary = {
            "case_id": case.case_id,
            "timestamp": case.created_at.isoformat(),
            "status": case.current_stage,
            
            "voice_analysis": None,
            "symptoms": None,
            "triage_decision": None,
            "care_plan_summary": None
        }
        
        if case.voice_result:
            summary["voice_analysis"] = {
                "transcript": case.voice_result.transcript,
                "urgency_level": case.voice_result.urgency_level,
                "urgency_score": case.voice_result.urgency_score
            }
        
        if case.symptom_result:
            summary["symptoms"] = {
                "count": len(case.symptom_result.symptoms),
                "primary_symptoms": [s.name for s in case.symptom_result.symptoms[:3]],
                "medical_history": case.symptom_result.medical_history,
                "allergies": case.symptom_result.allergies
            }
        
        if case.triage_decision:
            summary["triage_decision"] = {
                "urgency": case.triage_decision.urgency_level.value,
                "confidence": case.triage_decision.urgency_confidence,
                "disposition": case.triage_decision.disposition.value,
                "red_flags": len(case.triage_decision.red_flags),
                "recommended_tests": case.triage_decision.recommended_tests
            }
        
        if case.care_plan:
            summary["care_plan_summary"] = {
                "doctor_chief_complaint": case.care_plan.doctor_summary.chief_complaint,
                "patient_wait_time": case.care_plan.patient_instructions.estimated_wait_time,
                "system_actions": len(case.care_plan.system_actions.notifications)
            }
        
        return summary


# Demo/Testing
async def demo_orchestration():
    """Demo function showing full orchestration"""
    
    orchestrator = MediScanOrchestrator()
    
    # Simulated progress callback
    async def progress_update(case_id: str, stage: str, percentage: int):
        print(f"[PROGRESS] Case {case_id}: {stage} - {percentage}%")
    
    # Process a sample case
    case = await orchestrator.process_patient(
        case_id="DEMO-001",
        audio_file="sample_audio.wav",  # In real system
        document_images=["insurance_card.jpg"],  # In real system
        progress_callback=progress_update
    )
    
    # Export summary
    summary = orchestrator.export_case_summary("DEMO-001")
    print("\n" + "="*60)
    print("CASE SUMMARY")
    print("="*60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(demo_orchestration())