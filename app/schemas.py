from pydantic import BaseModel, Field
from typing import Dict, List


class SensorReading(BaseModel):
    temperature: float = Field(..., ge=0, le=150)
    vibration: float = Field(..., ge=0, le=10)
    pressure: float = Field(..., ge=0, le=20)
    rpm: float = Field(..., ge=0, le=10000)
    voltage: float = Field(..., ge=0, le=1000)
    current: float = Field(..., ge=0, le=100)
    operating_hours: float = Field(..., ge=0, le=200000)


class ExplanationItem(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str


class PredictionResponse(BaseModel):
    machine_status: str
    status_probability: float
    status_probabilities: Dict[str, float]

    health_score: float

    anomaly: bool
    anomaly_score: float
    anomaly_severity: str

    likely_fault: str
    maintenance_priority: str
    recommendation: str

    recommended_actions: List[str]
    contributing_factors: List[str]

    explanations: List[ExplanationItem]


class HistoryItem(BaseModel):
    id: int
    created_at: str
    machine_status: str
    status_probability: float
    health_score: float
    anomaly: bool
    anomaly_severity: str
    likely_fault: str
    maintenance_priority: str