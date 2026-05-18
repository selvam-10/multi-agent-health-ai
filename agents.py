from typing import List, Dict, Any
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class AgentBase:
    def __init__(self, name: str):
        self.name = name

    def act(self, payload: Any):
        raise NotImplementedError()


class MonitoringAgent(AgentBase):
    """Monitors real-time vitals and detects critical abnormalities."""
    def __init__(self, name: str = "monitoring"):
        super().__init__(name)

    def act(self, vitals_row: Dict[str, Any]) -> List[str]:
        alerts = []
        hr = vitals_row.get("heart_rate")
        sys = vitals_row.get("systolic_bp")
        dia = vitals_row.get("diastolic_bp")
        temp = vitals_row.get("temperature")
        chol = vitals_row.get("cholesterol")

        if hr is not None:
            if hr > 110:
                alerts.append("⚠️ TACHYCARDIA: Heart rate > 110 bpm")
            elif hr < 50:
                alerts.append("⚠️ BRADYCARDIA: Heart rate < 50 bpm")

        if sys is not None:
            if sys > 160:
                alerts.append("⚠️ HYPERTENSIVE CRISIS: Systolic BP > 160 mmHg")
            elif sys < 90:
                alerts.append("⚠️ HYPOTENSION: Systolic BP < 90 mmHg")

        if temp is not None and temp >= 38.0:
            alerts.append("🌡️ FEVER: Temperature >= 38°C")

        if chol is not None and chol > 300:
            alerts.append("⚠️ HIGH CHOLESTEROL: > 300 mg/dL")

        return alerts


class AnalysisAgent(AgentBase):
    """Performs statistical analysis on patient vitals."""
    def __init__(self, name: str = "analysis"):
        super().__init__(name)

    def act(self, patient_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not patient_data:
            return {}

        analysis = {}
        
        # Extract numeric fields
        hrs = [r.get("heart_rate") for r in patient_data if r.get("heart_rate") is not None]
        sys_bps = [r.get("systolic_bp") for r in patient_data if r.get("systolic_bp") is not None]
        chols = [r.get("cholesterol") for r in patient_data if r.get("cholesterol") is not None]

        if hrs:
            analysis["avg_heart_rate"] = round(sum(hrs) / len(hrs), 1)
            analysis["max_heart_rate"] = max(hrs)
            analysis["min_heart_rate"] = min(hrs)

        if sys_bps:
            analysis["avg_systolic_bp"] = round(sum(sys_bps) / len(sys_bps), 1)
            analysis["max_systolic_bp"] = max(sys_bps)

        if chols:
            analysis["avg_cholesterol"] = round(sum(chols) / len(chols), 1)

        return analysis


class AnomalyDetectionAgent(AgentBase):
    """Detects anomalies using statistical thresholds."""
    def __init__(self, name: str = "anomaly_detection"):
        super().__init__(name)

    def act(self, current: Dict[str, Any], stats: Dict[str, Any]) -> List[str]:
        anomalies = []
        
        avg_hr = stats.get("avg_heart_rate", 70)
        current_hr = current.get("heart_rate")
        if current_hr and abs(current_hr - avg_hr) > 30:
            anomalies.append(f"🔴 ANOMALY: HR deviation {current_hr} vs avg {avg_hr}")

        avg_sys = stats.get("avg_systolic_bp", 120)
        current_sys = current.get("systolic_bp")
        if current_sys and abs(current_sys - avg_sys) > 25:
            anomalies.append(f"🔴 ANOMALY: Systolic BP deviation {current_sys} vs avg {avg_sys}")

        return anomalies


class RiskPredictionAgent(AgentBase):
    """Predicts health risks based on vitals and history."""
    def __init__(self, name: str = "risk_prediction"):
        super().__init__(name)

    def act(self, vitals: Dict[str, Any], stats: Dict[str, Any]) -> Dict[str, Any]:
        risk_score = 0
        risk_factors = []

        # Check multiple risk indicators
        hr = vitals.get("heart_rate")
        sys = vitals.get("systolic_bp")
        chol = vitals.get("cholesterol")
        age = vitals.get("age")
        sex = vitals.get("sex")

        if hr and hr > 100:
            risk_score += 20
            risk_factors.append("Elevated heart rate")
        if sys and sys > 140:
            risk_score += 25
            risk_factors.append("Elevated blood pressure")
        if chol and chol > 240:
            risk_score += 20
            risk_factors.append("High cholesterol")
        if age and age > 55:
            risk_score += 15
            risk_factors.append("Advanced age")

        risk_level = "LOW" if risk_score < 30 else ("MEDIUM" if risk_score < 60 else "HIGH")

        return {
            "risk_score": min(risk_score, 100),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
        }


class GenerativeAgent(AgentBase):
    """Generates health summaries and clinical recommendations."""
    def __init__(self, model_name: str = "gpt2", device: int = -1, use_ml: bool = True):
        super().__init__("generative")
        self.model_name = model_name
        self.device = device
        self.use_ml = use_ml and HAS_TRANSFORMERS
        self._generator = None

    @property
    def generator(self):
        if self._generator is None and self.use_ml:
            try:
                self._generator = pipeline("text-generation", model=self.model_name, device=self.device)
            except Exception:
                self.use_ml = False
        return self._generator

    def _simple_summary(self, patient_id: str, vitals: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate summary without ML models (rule-based)."""
        hr = analysis.get('avg_heart_rate')
        sys_bp = analysis.get('avg_systolic_bp')
        chol = analysis.get('avg_cholesterol')
        
        summary_parts = []
        if hr:
            if hr > 100:
                summary_parts.append(f"Patient exhibits elevated resting heart rate ({hr} bpm). Consider cardiac workup.")
            elif hr < 60:
                summary_parts.append(f"Patient has low resting heart rate ({hr} bpm). Monitor for bradycardia.")
            else:
                summary_parts.append(f"Heart rate within normal range ({hr} bpm).")
        
        if sys_bp:
            if sys_bp > 140:
                summary_parts.append(f"Systolic BP elevated ({sys_bp} mmHg). Recommend BP management and lifestyle changes.")
            elif sys_bp < 100:
                summary_parts.append(f"Systolic BP low ({sys_bp} mmHg). Monitor for hypotension.")
            else:
                summary_parts.append(f"Blood pressure within normal range ({sys_bp} mmHg).")
        
        if chol:
            if chol > 240:
                summary_parts.append(f"Cholesterol elevated ({chol} mg/dL). Recommend lipid panel review and dietary intervention.")
            else:
                summary_parts.append(f"Cholesterol within acceptable range ({chol} mg/dL).")
        
        return " ".join(summary_parts) if summary_parts else "Patient vitals within acceptable range. Continue routine monitoring."

    def act(self, patient_id: str, vitals: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        # Use simple summary if ML not available
        if not self.use_ml or not self.generator:
            return self._simple_summary(patient_id, vitals, analysis)
        
        prompt = f"""Patient {patient_id} Health Summary:
- Average Heart Rate: {analysis.get('avg_heart_rate', 'N/A')} bpm
- Average Systolic BP: {analysis.get('avg_systolic_bp', 'N/A')} mmHg
- Average Cholesterol: {analysis.get('avg_cholesterol', 'N/A')} mg/dL

Provide a concise clinical summary and recommendations."""

        try:
            out = self.generator(prompt, max_length=100, do_sample=True, temperature=0.7, num_return_sequences=1)
            text = out[0]["generated_text"]
            summary = text.replace(prompt, "").strip()
            return summary if summary else self._simple_summary(patient_id, vitals, analysis)
        except Exception:
            return self._simple_summary(patient_id, vitals, analysis)


class AlertReportAgent(AgentBase):
    """Generates formal alerts and clinical reports."""
    def __init__(self, name: str = "alert_report"):
        super().__init__(name)

    def act(self, patient_id: str, alerts: List[str], anomalies: List[str], 
            risk_data: Dict[str, Any], summary: str) -> Dict[str, Any]:
        
        severity = "LOW"
        if risk_data.get("risk_level") == "HIGH":
            severity = "CRITICAL"
        elif risk_data.get("risk_level") == "MEDIUM" or alerts:
            severity = "HIGH"

        report = {
            "patient_id": patient_id,
            "severity": severity,
            "alerts": alerts,
            "anomalies": anomalies,
            "risk_assessment": risk_data,
            "clinical_summary": summary,
            "recommendation": "Schedule follow-up with physician" if severity != "LOW" else "Continue routine monitoring"
        }

        return report
