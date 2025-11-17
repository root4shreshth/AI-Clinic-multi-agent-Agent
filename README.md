# 🏥 Voice-Driven Medical Triage System

**An AI-powered symptom intake and triage assistant using ADK + Gemini + Vertex AI**

---

## 🚀 Quick Start

### 1. **Clone & Setup**

```bash
# create python virtual environment
python -m venv .venv

# activate the virtual environment
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. **Run Demo**

```bash
python orchestrator/agent_coordinator.py
```

---

## 🎯 How It Works

### **Multi-Agent Workflow**

```
1. Voice Input / Text Input
   ↓
2. Transcript → Symptom Extraction Agent (Gemini)
   ↓
3. Structured Symptoms → Triage Agent (Classification)
   ↓
4. Urgency Level → Summary Generator Agent
   ↓
5. Final Output → Doctor-Ready Report
```

### **Agent Roles**

| Agent | Purpose | Technology |
|-------|---------|------------|
| **Coordinator** | Orchestrates workflow | ADK |
| **Voice Processor** | Converts speech to text | Google Cloud Speech API |
| **Symptom Extractor** | Extracts structured symptoms | Gemini 1.5 Pro |
| **Triage Agent** | Classifies urgency (LOW/MEDIUM/HIGH) | Gemini + Rule Engine |
| **Summary Generator** | Creates clinical summaries | Gemini 1.5 Pro |
