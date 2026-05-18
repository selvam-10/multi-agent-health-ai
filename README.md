# Multi-Agent Generative AI Framework for Patient Health Tracking

A comprehensive multi-agent AI system for real-time patient health monitoring, analysis, and clinical decision support. The framework processes patient vitals and medical data using specialized AI agents to detect anomalies, predict risks, and generate clinical recommendations.

## Architecture

```
Patient Data (Vitals, Sensors, Medical History)
        ↓
   Data Preprocessing (Column Normalization)
        ↓
Multi-Agent AI Framework:
  ├─ Monitoring Agent      → Real-time vital sign monitoring
  ├─ Analysis Agent        → Statistical analysis of patient trends
  ├─ Anomaly Detection     → Detects deviations from baseline
  ├─ Risk Prediction Agent → Calculates health risk scores
  └─ Generative AI Agent   → Generates clinical summaries & recommendations
        ↓
  Alert & Report Agent    → Creates formal alerts and reports
        ↓
  Healthcare Dashboard    → Doctor/Caregiver Interface
```

## Components

### 1. **Monitoring Agent**
- Detects critical abnormalities in real-time
- Alerts on: tachycardia, bradycardia, hypertension, hypotension, fever, high cholesterol
- Returns structured alert list

### 2. **Analysis Agent**
- Computes aggregate statistics on patient data
- Calculates: average/max/min heart rate, blood pressure, cholesterol
- Used as baseline for anomaly detection

### 3. **Anomaly Detection Agent**
- Identifies deviations from patient's historical baseline
- Flags unusual patterns for clinical review
- Statistical threshold-based (±30 bpm for HR, ±25 mmHg for BP)

### 4. **Risk Prediction Agent**
- Scores patient health risk (0-100 scale)
- Risk levels: LOW (<30), MEDIUM (30-60), HIGH (>60)
- Identifies contributing risk factors (age, vitals, cholesterol, etc.)

### 5. **Generative AI Agent**
- Uses transformer-based LLM (GPT-2 by default)
- Generates natural language clinical summaries
- Synthesizes analysis from other agents

### 6. **Alert & Report Agent**
- Consolidates all agent outputs into formal clinical report
- Assigns severity (LOW / HIGH / CRITICAL)
- Provides actionable recommendations

## Output

The system generates two output formats:

### Dashboard Format (Console Display)
```
================================================================================
PATIENT HEALTH DASHBOARD | Patient ID: row_0
================================================================================

🟠 SEVERITY: HIGH

RISK ASSESSMENT:
  Risk Level: MEDIUM
  Risk Score: 40/100
  Risk Factors: Elevated heart rate, High cholesterol

ACTIVE ALERTS:
  ⚠️ TACHYCARDIA: Heart rate > 110 bpm

CLINICAL SUMMARY:
  [AI-generated clinical summary]

RECOMMENDATION: Schedule follow-up with physician
================================================================================
```

### JSON Format (Structured Data)
```json
{
  "patient_id": "row_0",
  "severity": "HIGH",
  "alerts": ["⚠️ TACHYCARDIA: Heart rate > 110 bpm"],
  "anomalies": [],
  "risk_assessment": {
    "risk_score": 40,
    "risk_level": "MEDIUM",
    "risk_factors": ["Elevated heart rate", "High cholesterol"]
  },
  "clinical_summary": "...",
  "recommendation": "Schedule follow-up with physician"
}
```

## Quick Start

### Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option 1: Web Dashboard (Recommended) 🌐
```powershell
.\start_dashboard.ps1
# Opens at http://localhost:8501
```

Or manually:
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend.py
```

### Option 2: Command Line
```powershell
# Dashboard output (default)
python run_demo.py --data dataset.csv --patients 3

# JSON output
python run_demo.py --data dataset.csv --patients 3 --output json

# Custom settings
python run_demo.py --data dataset.csv --patients 5 --model distilgpt2
```

### Arguments
- `--data`: Path to patient dataset CSV (default: `dataset.csv`)
- `--model`: HuggingFace model name for text generation (default: `gpt2`)
- `--patients`: Number of patients to process (default: 3)
- `--output`: Output format (`dashboard` or `json`; default: `dashboard`)

## Web Dashboard Features

**The Streamlit dashboard provides:**
- 📊 **Dashboard Tab**: Real-time patient metrics and severity tracking
- 🚀 **Run Analysis Tab**: Generate reports from any dataset with live progress
- 📋 **View Reports Tab**: Browse, filter, and export patient reports
- 📈 **Analytics Tab**: Population statistics and trend visualization

See [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md) for complete dashboard documentation.

## Project Structure
```
d:\Final yr project\
├── run_demo.py              # Main demo runner
├── dataset.csv              # Sample patient data
├── requirements.txt         # Python dependencies
├── patient_reports.json     # Output reports
└── multi_agent/
    ├── __init__.py
    ├── data.py              # Data loading & preprocessing
    └── agents.py            # All agent implementations
```

## Dataset Format

Your `dataset.csv` should contain patient vitals. Required columns (will be auto-normalized):
- `age`, `sex` (demographics)
- Heart rate variants: `max_heart_rate`, `heart_rate`, `hr`
- Blood pressure variants: `resting_bp_s`, `systolic_bp`
- `cholesterol`, `temperature` (optional)
- `target` (optional; outcome/diagnosis)

Example:
```csv
age,sex,resting_bp_s,max_heart_rate,cholesterol,target
40.0,1.0,140.0,172.0,289.0,0.0
49.0,0.0,160.0,156.0,180.0,1.0
```

## Customization

### Add More Agents
Edit `multi_agent/agents.py` to add new agent classes inheriting from `AgentBase`:
```python
class CustomAgent(AgentBase):
    def __init__(self):
        super().__init__("custom_agent")
    
    def act(self, payload):
        # Your logic here
        return results
```

### Adjust Alert Thresholds
Edit `MonitoringAgent.act()` in `multi_agent/agents.py` to change vital sign thresholds.

### Use Different Models
Replace `gpt2` with any HuggingFace model:
```powershell
python run_demo.py --data dataset.csv --model distilgpt2 --patients 3
python run_demo.py --data dataset.csv --model EleutherAI/gpt-neo-125M --patients 3
```

## Clinical Disclaimer

⚠️ **This is a prototype framework for educational purposes.** For real clinical deployment:
1. Use medically-validated AI models
2. Implement HIPAA/privacy compliance
3. Add authentication & access control
4. Integrate with EHR systems
5. Conduct rigorous clinical validation
6. Get regulatory approval (FDA, etc.)

## Future Enhancements

- [ ] Web dashboard UI (Flask/React)
- [ ] Real-time data streaming (Kafka, WebSocket)
- [ ] Multi-modal input (EKG, imaging)
- [ ] Federated learning for privacy
- [ ] Integration with wearable devices
- [ ] Causal inference for recommendations
- [ ] Uncertainty quantification

## License & Attribution

Educational use only. For medical applications, consult healthcare professionals.

## Support

Questions or issues? Check the logs or adjust model/threshold parameters.
\n