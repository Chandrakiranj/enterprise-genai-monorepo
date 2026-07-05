import re
from pydantic import BaseModel, Field

class SafetyAssessment(BaseModel):
    is_safe: bool = Field(description="True if the payload passes architectural compliance validation rules.")
    risk_category: str = Field(default="None", description="Identified threat class or adversarial injection category.")

class BoundaryGuardrail:
    def __init__(self, config: dict):
        self.guard_id = config['safety_configuration']['guard_model_id']
        self.confidence = config['safety_configuration']['confidence_threshold']
        self.malicious_injection_patterns = [
            r"ignore\s+previous\s+instructions",
            r"system\s+prompt\s+extraction",
            r"drop\s+table",
            r"bypass\s+guardrails",
            r"reveal\s+internal\s+secrets"
        ]

    def screen_input_payload(self, user_query: str) -> SafetyAssessment:
        normalized_query = user_query.lower().strip()
        for pattern in self.malicious_injection_patterns:
            if re.search(pattern, normalized_query):
                return SafetyAssessment(
                    is_safe=False, 
                    risk_category=f"Adversarial Attack Vector: Found match for pattern '{pattern}'"
                )
        return SafetyAssessment(is_safe=True, risk_category="None")

    def screen_output_payload(self, model_response: str) -> SafetyAssessment:
        normalized_response = model_response.lower()
        if "internal_token_secret" in normalized_response or "aws_secret_access_key" in normalized_response:
            return SafetyAssessment(is_safe=False, risk_category="Data Leakage Violation: Protected Server Tokens Found")
        return SafetyAssessment(is_safe=True, risk_category="None")
