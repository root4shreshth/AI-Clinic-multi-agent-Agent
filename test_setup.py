from agents.symptom_reasoning_agent import SymptomReasoningAgent

agent = SymptomReasoningAgent()
result = agent.analyze_symptoms(
    "I have chest pain for 2 days",
    {"urgency_level": "high"}
)
print(f"✅ Found {len(result.symptoms)} symptoms")