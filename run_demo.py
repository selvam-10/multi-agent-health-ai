from multi_agent.data import load_data, get_patient_groups
from multi_agent.agents import (
    MonitoringAgent,
    AnalysisAgent,
    AnomalyDetectionAgent,
    RiskPredictionAgent,
    GenerativeAgent,
    AlertReportAgent,
)
import argparse
import json


def format_dashboard_report(report: dict) -> str:
    """Format a patient report as a healthcare dashboard display."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"PATIENT HEALTH DASHBOARD | Patient ID: {report['patient_id']}")
    lines.append("=" * 80)
    
    # Severity badge
    severity = report['severity']
    severity_emoji = "🔴" if severity == "CRITICAL" else "🟠" if severity == "HIGH" else "🟢"
    lines.append(f"\n{severity_emoji} SEVERITY: {severity}\n")
    
    # Risk Assessment
    risk = report['risk_assessment']
    lines.append(f"RISK ASSESSMENT:")
    lines.append(f"  Risk Level: {risk.get('risk_level', 'N/A')}")
    lines.append(f"  Risk Score: {risk.get('risk_score', 0)}/100")
    if risk.get('risk_factors'):
        lines.append(f"  Risk Factors: {', '.join(risk['risk_factors'])}")
    lines.append("")
    
    # Alerts
    if report['alerts']:
        lines.append("ACTIVE ALERTS:")
        for alert in report['alerts']:
            lines.append(f"  {alert}")
        lines.append("")
    
    # Anomalies
    if report['anomalies']:
        lines.append("DETECTED ANOMALIES:")
        for anomaly in report['anomalies']:
            lines.append(f"  {anomaly}")
        lines.append("")
    
    # Clinical Summary
    lines.append("CLINICAL SUMMARY:")
    lines.append(f"  {report['clinical_summary']}")
    lines.append("")
    
    # Recommendation
    lines.append(f"RECOMMENDATION: {report['recommendation']}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main(path: str, model: str, patients: int, output_format: str = "dashboard", use_ml: bool = False):
    print(f"\n📊 Loading patient data from {path}...")
    df = load_data(path)
    groups = get_patient_groups(df)
    
    # Initialize all agents
    monitoring = MonitoringAgent()
    analysis = AnalysisAgent()
    anomaly_det = AnomalyDetectionAgent()
    risk_pred = RiskPredictionAgent()
    generative = GenerativeAgent(model_name=model, use_ml=use_ml)
    alert_report = AlertReportAgent()
    
    all_reports = []
    printed = 0
    
    for patient_id, patient_df in groups.items():
        if printed >= patients:
            break
        
        # Convert to list of dicts
        patient_data = patient_df.to_dict(orient="records")
        if not patient_data:
            continue
        
        latest = patient_data[-1]  # most recent reading
        
        # Run all agents
        alerts = monitoring.act(latest)
        stats = analysis.act(patient_data)
        anomalies = anomaly_det.act(latest, stats)
        risk_data = risk_pred.act(latest, stats)
        summary = generative.act(patient_id, latest, stats)
        
        report = alert_report.act(patient_id, alerts, anomalies, risk_data, summary)
        all_reports.append(report)
        
        # Display report
        if output_format == "dashboard":
            print(format_dashboard_report(report))
        elif output_format == "json":
            print(json.dumps(report, indent=2))
        
        printed += 1
    
    # Save all reports to file
    with open("patient_reports.json", "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\n✅ Reports saved to patient_reports.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name for generation")
    parser.add_argument("--patients", type=int, default=3, help="How many patients to demo")
    parser.add_argument("--output", default="dashboard", choices=["dashboard", "json"], help="Output format")
    parser.add_argument("--ml", action="store_true", help="Use ML models (requires transformers)")
    args = parser.parse_args()
    main(args.data, args.model, args.patients, args.output, use_ml=args.ml)
