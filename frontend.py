"""
Streamlit Web Dashboard for Multi-Agent Patient Health Tracking System
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from data import load_data, get_patient_groups
from agents import (
    MonitoringAgent,
    AnalysisAgent,
    AnomalyDetectionAgent,
    RiskPredictionAgent,
    GenerativeAgent,
    AlertReportAgent,
)

# Try to load ML model dependencies
try:
    import joblib
    HAS_ML = True
except ImportError:
    HAS_ML = False

# Set page configuration
st.set_page_config(
    page_title="Patient Health Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .metric-card {
        padding: 15px;
        border-radius: 8px;
        background: #f8f9fa;
        border-left: 4px solid #2c3e50;
        text-align: left;
    }
    .alert-critical {
        padding: 12px;
        background: #fee;
        border-left: 4px solid #d32f2f;
        border-radius: 4px;
    }
    .alert-high {
        padding: 12px;
        background: #fff3cd;
        border-left: 4px solid #ff9800;
        border-radius: 4px;
    }
    .alert-medium {
        padding: 12px;
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        border-radius: 4px;
    }
    .alert-low {
        padding: 12px;
        background: #f1f8e9;
        border-left: 4px solid #4caf50;
        border-radius: 4px;
    }
    .header-section {
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Patient Health Monitoring Dashboard")
st.markdown("<div class='header-section'>Multi-Agent AI Framework for Clinical Analysis</div>", unsafe_allow_html=True)

# Create main tabs for all features
main_tabs = st.tabs(["Live Monitoring", "Patient Dashboard", "Run Analysis", "Reports", "Analytics"])

def load_reports():
    """Load patient reports from JSON file."""
    report_file = Path("patient_reports.json")
    if report_file.exists():
        with open(report_file) as f:
            return json.load(f)
    return []

def get_severity_color(severity):
    """Get color for severity level."""
    colors = {
        "CRITICAL": "#ff4444",
        "HIGH": "#ff9800",
        "MEDIUM": "#ffc107",
        "LOW": "#4caf50"
    }
    return colors.get(severity, "#9c27b0")


@st.cache_resource
def load_ml_model():
    """Load the trained ML model and scaler."""
    try:
        model = joblib.load("model.joblib")
        scaler = joblib.load("scaler.joblib")
        feature_names = joblib.load("feature_names.joblib")
        return model, scaler, feature_names
    except FileNotFoundError:
        return None, None, None


def generate_random_patient_data(num_samples: int = 1, feature_names: list = None):
    """Generate random patient data based on realistic medical ranges."""
    if feature_names is None:
        feature_names = ['age', 'sex', 'chest_pain_type', 'resting_bp_s', 'cholesterol', 
                        'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 'exercise_angina', 
                        'oldpeak', 'st_slope']
    
    ranges = {
        'age': (29, 76),
        'sex': (0, 1),
        'chest_pain_type': (0, 3),
        'resting_bp_s': (90, 200),
        'cholesterol': (0, 600),
        'fasting_blood_sugar': (0, 1),
        'resting_ecg': (0, 2),
        'max_heart_rate': (60, 220),
        'exercise_angina': (0, 1),
        'oldpeak': (0, 6.2),
        'st_slope': (0, 2),
    }
    
    random_data = []
    for _ in range(num_samples):
        sample = {}
        for feature in feature_names:
            if feature in ranges:
                low, high = ranges[feature]
                if feature in ['sex', 'fasting_blood_sugar', 'exercise_angina', 'chest_pain_type', 'resting_ecg', 'st_slope']:
                    sample[feature] = float(np.random.randint(int(low), int(high) + 1))
                else:
                    sample[feature] = float(np.random.uniform(low, high))
        random_data.append(sample)
    
    return pd.DataFrame(random_data)


def format_dataframe_for_display(df: pd.DataFrame):
    """Format dataframe for display with capitalized columns and readable values."""
    display_df = df.copy()
    
    # Capitalize column names
    display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
    
    # Convert sex from 0/1 to Female/Male
    if 'Sex' in display_df.columns:
        display_df['Sex'] = display_df['Sex'].apply(lambda x: 'Female' if x == 0 else 'Male')
    
    # Round numeric columns for readability
    for col in display_df.columns:
        if display_df[col].dtype in ['float64', 'float32']:
            display_df[col] = display_df[col].apply(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)
    
    return display_df


def predict_heart_disease(patient_df: pd.DataFrame, model, scaler, feature_names):
    """Make heart disease predictions on patient data."""
    X_scaled = scaler.transform(patient_df)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    return predictions, probabilities


def create_doctor_alert(risk_percentage: float, patient_id: str, patient_data: dict) -> dict:
    """Create a doctor alert for high-risk patients."""
    alert = {
        "patient_id": patient_id,
        "timestamp": datetime.now().isoformat(),
        "risk_percentage": risk_percentage,
        "risk_level": "🔴 CRITICAL" if risk_percentage >= 80 else ("🟠 HIGH" if risk_percentage >= 60 else "🟡 MEDIUM"),
        "vitals": {
            "age": patient_data.get('age'),
            "resting_bp": patient_data.get('resting_bp_s'),
            "max_heart_rate": patient_data.get('max_heart_rate'),
            "cholesterol": patient_data.get('cholesterol'),
        }
    }
    return alert


def process_patient_through_ai_agents(patient_data: dict, patient_id: str):
    """Process patient data through all AI agents for comprehensive analysis."""
    try:
        # Initialize all agents
        monitoring = MonitoringAgent()
        analysis = AnalysisAgent()
        anomaly_det = AnomalyDetectionAgent()
        risk_pred = RiskPredictionAgent()
        generative = GenerativeAgent(use_ml=False)  # Use rule-based to avoid ML model loading
        alert_report = AlertReportAgent()
        
        # Convert to list for analysis
        patient_list = [patient_data]
        
        # Run agents
        monitoring_alerts = monitoring.act(patient_data)
        stats = analysis.act(patient_list)
        anomalies = anomaly_det.act(patient_data, stats)
        risk_data = risk_pred.act(patient_data, stats)
        clinical_summary = generative.act(patient_id, patient_data, stats)
        
        # Generate alert report
        report = alert_report.act(patient_id, monitoring_alerts, anomalies, risk_data, clinical_summary)
        
        return {
            "monitoring_alerts": monitoring_alerts,
            "anomalies": anomalies,
            "risk_assessment": risk_data,
            "clinical_summary": clinical_summary,
            "full_report": report
        }
    except Exception as e:
        return {
            "monitoring_alerts": [],
            "anomalies": [],
            "risk_assessment": {},
            "clinical_summary": f"Error generating analysis: {str(e)}",
            "full_report": None,
            "error": str(e)
        }

def display_patient_card(report):
    """Display a formatted patient card."""
    severity = report.get("severity", "UNKNOWN")
    patient_id = report.get("patient_id", "N/A")
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Patient ID",
                patient_id,
                delta=None
            )
        
        with col2:
            severity_emoji = "🔴" if severity == "CRITICAL" else "🟠" if severity == "HIGH" else "🟡" if severity == "MEDIUM" else "🟢"
            st.metric(
                "Severity",
                f"{severity_emoji} {severity}",
                delta=None
            )
        
        with col3:
            risk_score = report.get("risk_assessment", {}).get("risk_score", 0)
            st.metric(
                "Risk Score",
                f"{risk_score}/100",
                delta=None
            )
        
        with col4:
            alerts_count = len(report.get("alerts", []))
            st.metric(
                "Active Alerts",
                alerts_count,
                delta=None
            )

def display_alert_details(report):
    """Display detailed alert information."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Active Alerts")
        alerts = report.get("alerts", [])
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.success("No active alerts")
    
    with col2:
        st.subheader("🔍 Anomalies Detected")
        anomalies = report.get("anomalies", [])
        if anomalies:
            for anomaly in anomalies:
                st.error(anomaly)
        else:
            st.info("No anomalies detected")

def display_risk_assessment(report):
    """Display risk assessment details."""
    risk = report.get("risk_assessment", {})
    
    st.subheader("📊 Risk Assessment")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Risk Level",
            risk.get("risk_level", "N/A"),
            delta=None
        )
    
    with col2:
        st.metric(
            "Risk Score",
            f"{risk.get('risk_score', 0)}/100",
            delta=None
        )
    
    with col3:
        st.metric(
            "Contributing Factors",
            len(risk.get("risk_factors", [])),
            delta=None
        )
    
    # Risk factors list
    if risk.get("risk_factors"):
        st.markdown("**Risk Factors:**")
        for factor in risk["risk_factors"]:
            st.markdown(f"- {factor}")

def display_clinical_summary(report):
    """Display clinical summary and recommendations."""
    st.subheader("📋 Clinical Summary")
    summary = report.get("clinical_summary", "No summary available")
    st.info(summary)
    
    st.subheader("💡 Recommendation")
    recommendation = report.get("recommendation", "Continue routine monitoring")
    st.success(recommendation)

# Page: Dashboard
# Page: Patient Dashboard Tab
with main_tabs[1]:
    st.header("Patient Dashboard")
    
    reports = load_reports()
    
    if not reports:
        st.warning("No reports found. Run analysis to generate reports.")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        critical = sum(1 for r in reports if r.get("severity") == "CRITICAL")
        high = sum(1 for r in reports if r.get("severity") == "HIGH")
        medium = sum(1 for r in reports if r.get("severity") == "MEDIUM")
        low = sum(1 for r in reports if r.get("severity") == "LOW")
        
        with col1:
            st.metric("Critical", critical)
        with col2:
            st.metric("High", high)
        with col3:
            st.metric("Medium", medium)
        with col4:
            st.metric("Low", low)
        
        st.divider()
        
        # Patient selection
        patient_options = [r.get("patient_id", "Unknown") for r in reports]
        selected_patient = st.selectbox("Select Patient", patient_options)
        
        # Get selected patient report
        patient_report = next((r for r in reports if r.get("patient_id") == selected_patient), None)
        
        if patient_report:
            display_patient_card(patient_report)
            st.divider()
            
            # Tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["Alerts", "Risk", "Summary", "Full Report"])
            
            with tab1:
                display_alert_details(patient_report)
            
            with tab2:
                display_risk_assessment(patient_report)
            
            with tab3:
                display_clinical_summary(patient_report)
            
            with tab4:
                st.json(patient_report)

# Page: Live Monitoring Tab
with main_tabs[0]:
    st.header("Live Patient Monitoring & Predictions")
    st.markdown("Generate real-time patient data for AI-driven clinical analysis")
    
    # Display auto-refresh status at the top
    if st.session_state.get('auto_refresh_enabled', False):
        col_status1, col_status2, col_status3 = st.columns([1, 2, 1])
        
        with col_status1:
            st.markdown(f"""
            <div style='padding: 15px; background: #e8f5e9; border-left: 4px solid #4caf50; border-radius: 4px;'>
                <h3 style='margin: 0; color: #2e7d32;'>AUTO-REFRESH ACTIVE</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col_status2:
            countdown = st.session_state.get('countdown', 5)
            st.markdown(f"""
            <div style='padding: 15px; background: #fff3cd; border-left: 4px solid #ff9800; border-radius: 4px; text-align: center;'>
                <h2 style='margin: 0; color: #ff6f00;'>Next Refresh: {countdown}s</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col_status3:
            if st.button("Stop Auto-Refresh", use_container_width=True, key="stop_refresh_top"):
                st.session_state.auto_refresh_enabled = False
                st.session_state.auto_refresh_checkbox = False
                st.rerun()
        
        st.divider()
    
    # Display active notifications
    if 'notifications' in st.session_state and st.session_state.notifications:
        st.divider()
        st.subheader("Active Notifications")
        
        # Display notifications in a grid
        for idx, notification in enumerate(st.session_state.notifications):
            col_notif1, col_notif2, col_notif3 = st.columns([1, 3, 1])
            
            with col_notif1:
                st.markdown(f"<div style='padding: 12px; background: #fee; border-left: 4px solid #d32f2f; border-radius: 4px; text-align: center;'><h4 style='margin: 0;'>{notification['risk_level']}</h4></div>", unsafe_allow_html=True)
            
            with col_notif2:
                st.info(f"""
                **Patient:** {notification['patient_id']}  
                **Risk Score:** {notification['risk_percentage']:.1f}%  
                **Age:** {notification['vitals']['age']:.0f} years  
                **BP:** {notification['vitals']['resting_bp']:.0f} mmHg  
                **HR:** {notification['vitals']['max_heart_rate']:.0f} bpm
                """)
            
            with col_notif3:
                if st.button("Dismiss", key=f"dismiss_{idx}"):
                    st.session_state.notifications.pop(idx)
                    st.rerun()
        
        if st.button("Clear All Notifications"):
            st.session_state.notifications = []
            st.rerun()
        
        st.divider()
    
    if not HAS_ML:
        st.error("Required ML libraries not installed. Please install scikit-learn and joblib.")
    else:
        # Check if model exists
        model, scaler, feature_names = load_ml_model()
        
        if model is None:
            st.warning("⚠️ Trained model not found. Please train the model first using the 'Train Model' option below.")
            
            if st.button("📚 Train Model on Dataset"):
                with st.spinner("Training heart disease prediction model..."):
                    try:
                        from sklearn.model_selection import train_test_split
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.preprocessing import StandardScaler
                        
                        # Load and prepare data
                        df = pd.read_csv("dataset.csv")
                        X = df.drop('target', axis=1)
                        y = df['target']
                        feature_names = X.columns.tolist()
                        
                        # Split and scale
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Train model
                        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        model.fit(X_train_scaled, y_train)
                        
                        # Save model
                        joblib.dump(model, "model.joblib")
                        joblib.dump(scaler, "scaler.joblib")
                        joblib.dump(feature_names, "feature_names.joblib")
                        
                        train_score = model.score(X_train_scaled, y_train)
                        test_score = model.score(X_test_scaled, y_test)
                        
                        st.success("Model trained successfully!")
                        col_metric1, col_metric2 = st.columns(2)
                        with col_metric1:
                            st.metric("Training Accuracy", f"{train_score:.2%}")
                        with col_metric2:
                            st.metric("Testing Accuracy", f"{test_score:.2%}")
                        
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        else:
            # Live data generation controls
            st.divider()
            st.subheader("Generate Live Patient Data")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                num_patients = st.slider("Number of random patients to generate", 1, 20, 5)
            
            with col2:
                refresh_button = st.button("Generate New Data", use_container_width=True)
            
            with col3:
                # Bind checkbox to session state
                auto_refresh = st.checkbox(
                    "Auto-refresh (5s interval)", 
                    value=st.session_state.get('auto_refresh_enabled', False),
                    key='auto_refresh_checkbox'
                )
                
                # Update session state when checkbox changes
                if auto_refresh and not st.session_state.get('auto_refresh_enabled', False):
                    st.session_state.auto_refresh_enabled = True
                elif not auto_refresh:
                    st.session_state.auto_refresh_enabled = False
            
            if refresh_button or 'generated_data' not in st.session_state:
                # Generate random data
                st.session_state.generated_data = generate_random_patient_data(num_patients, feature_names)
                st.session_state.predictions = None
                st.session_state.alerts = []
                st.session_state.ai_agent_reports = {}
            
            # Display generated data
            if 'generated_data' in st.session_state and st.session_state.generated_data is not None:
                generated_df = st.session_state.generated_data.copy()
                display_df = format_dataframe_for_display(generated_df)
                display_df.index = display_df.index + 1
                
                st.subheader("Generated Patient Data")
                st.dataframe(display_df, use_container_width=True)
                
                # Make predictions using original numeric data
                st.divider()
                if st.button("Process Predictions", use_container_width=True):
                    with st.spinner("Making predictions..."):
                        try:
                            # Use original df for predictions (numeric sex values)
                            generated_df_numeric = st.session_state.generated_data.copy()
                            predictions, probabilities = predict_heart_disease(generated_df_numeric, model, scaler, feature_names)
                            st.session_state.predictions = predictions
                            st.session_state.probabilities = probabilities
                            
                            # Generate alerts for high-risk patients with AI agent analysis
                            st.session_state.alerts = []
                            st.session_state.ai_agent_reports = {}
                            
                            # Initialize notifications if not exists
                            if 'notifications' not in st.session_state:
                                st.session_state.notifications = []
                            
                            with st.spinner("Running AI agents for comprehensive analysis..."):
                                for idx, (prob, pred) in enumerate(zip(probabilities, predictions)):
                                    risk_pct = prob[1] * 100
                                    patient_id = f"Patient_{idx+1}"
                                    patient_dict = generated_df.iloc[idx].to_dict()
                                    
                                    if risk_pct >= 60:  # Alert threshold
                                        alert = create_doctor_alert(risk_pct, patient_id, patient_dict)
                                        st.session_state.alerts.append(alert)
                                        
                                        # Run AI agents for this patient
                                        ai_analysis = process_patient_through_ai_agents(patient_dict, patient_id)
                                        st.session_state.ai_agent_reports[patient_id] = ai_analysis
                                        
                                        # AUTOMATICALLY ADD NOTIFICATION
                                        notification = {
                                            "patient_id": patient_id,
                                            "risk_level": alert['risk_level'],
                                            "risk_percentage": alert['risk_percentage'],
                                            "timestamp": datetime.now().isoformat(),
                                            "vitals": alert['vitals'],
                                            "ai_report": ai_analysis
                                        }
                                        st.session_state.notifications.append(notification)
                            
                            st.success(f"Predictions completed. {len(st.session_state.alerts)} high-risk patients detected and notified!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error making predictions: {str(e)}")
                
                # Display predictions
                if st.session_state.predictions is not None:
                    st.divider()
                    st.subheader("Heart Disease Predictions")
                    
                    prediction_results = []
                    for idx, (row_data, pred, prob) in enumerate(zip(
                        st.session_state.generated_data.to_dict('records'),
                        st.session_state.predictions,
                        st.session_state.probabilities
                    )):
                        risk_pct = prob[1] * 100
                        sex_display = 'Female' if row_data['sex'] == 0 else 'Male'
                        prediction_results.append({
                            "Patient": f"Patient_{idx+1}",
                            "Heart Disease Risk (%)": round(risk_pct, 2),
                            "Status": "🔴 Disease Risk" if pred == 1 else "🟢 No Disease",
                            "Sex": sex_display,
                            "Age": f"{row_data['age']:.0f}",
                            "Resting BP": f"{row_data['resting_bp_s']:.0f}",
                            "Max HR": f"{row_data['max_heart_rate']:.0f}",
                            "Cholesterol": f"{row_data['cholesterol']:.0f}",
                        })
                    
                    predictions_df = pd.DataFrame(prediction_results)
                    predictions_df.index = predictions_df.index + 1
                    st.dataframe(predictions_df, use_container_width=True)
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Risk distribution
                        fig_risk = px.histogram(
                            st.session_state.probabilities[:, 1] * 100,
                            nbins=10,
                            title="Risk Score Distribution",
                            labels={"value": "Risk Score %", "count": "Number of Patients"},
                            color_discrete_sequence=["#ff6b6b"]
                        )
                        st.plotly_chart(fig_risk, use_container_width=True)
                    
                    with col2:
                        # High risk vs Low risk pie chart
                        high_risk = sum(1 for p in st.session_state.probabilities[:, 1] if p >= 0.6)
                        low_risk = len(st.session_state.probabilities) - high_risk
                        
                        fig_pie = px.pie(
                            values=[high_risk, low_risk],
                            names=["High Risk (60%+)", "Low Risk (<60%)"],
                            title="Patient Risk Distribution",
                            color_discrete_map={"High Risk (60%+)": "#ff6b6b", "Low Risk (<60%)": "#51cf66"}
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Doctor Alerts with AI Agent Analysis
                    if st.session_state.alerts:
                        st.divider()
                        st.subheader("Clinical Alerts - AI Analysis")
                        
                        for alert in st.session_state.alerts:
                            patient_id = alert['patient_id']
                            ai_report = st.session_state.ai_agent_reports.get(patient_id, {})
                            
                            # Create expandable alert card
                            with st.expander(f"{alert['risk_level']} | {patient_id} - Risk: {alert['risk_percentage']:.1f}%", expanded=True):
                                
                                # Top row: Risk level and vitals
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    st.markdown(f"### {alert['risk_level']}")
                                    st.metric("Risk Score", f"{alert['risk_percentage']:.1f}%")
                                
                                with col2:
                                    st.warning(f"""
                                    **Patient Vitals:**
                                    - Age: {alert['vitals']['age']:.0f} years
                                    - BP (Systolic): {alert['vitals']['resting_bp']:.0f} mmHg
                                    - Max Heart Rate: {alert['vitals']['max_heart_rate']:.0f} bpm
                                    - Cholesterol: {alert['vitals']['cholesterol']:.0f} mg/dL
                                    """)
                                
                                st.divider()
                                
                                # AI Agent Analysis Tabs
                                ai_tab1, ai_tab2, ai_tab3, ai_tab4, ai_tab5 = st.tabs(
                                    ["🔍 Monitoring", "⚠️ Anomalies", "📊 Risk", "📋 Summary", "📄 Full Report"]
                                )
                                
                                # Monitoring Alerts
                                with ai_tab1:
                                    monitoring_alerts = ai_report.get("monitoring_alerts", [])
                                    if monitoring_alerts:
                                        for m_alert in monitoring_alerts:
                                            st.warning(m_alert)
                                    else:
                                        st.info("✅ No critical monitoring alerts detected")
                                
                                # Anomalies
                                with ai_tab2:
                                    anomalies = ai_report.get("anomalies", [])
                                    if anomalies:
                                        for anomaly in anomalies:
                                            st.error(anomaly)
                                    else:
                                        st.info("✅ No anomalies detected")
                                
                                # Risk Assessment
                                with ai_tab3:
                                    risk_assessment = ai_report.get("risk_assessment", {})
                                    if risk_assessment:
                                        col_r1, col_r2, col_r3 = st.columns(3)
                                        
                                        with col_r1:
                                            st.metric(
                                                "Risk Level",
                                                risk_assessment.get("risk_level", "N/A")
                                            )
                                        
                                        with col_r2:
                                            st.metric(
                                                "Risk Score",
                                                f"{risk_assessment.get('risk_score', 0)}/100"
                                            )
                                        
                                        with col_r3:
                                            st.metric(
                                                "Risk Factors",
                                                len(risk_assessment.get("risk_factors", []))
                                            )
                                        
                                        if risk_assessment.get("risk_factors"):
                                            st.markdown("**Contributing Risk Factors:**")
                                            for factor in risk_assessment["risk_factors"]:
                                                st.markdown(f"- {factor}")
                                    else:
                                        st.info("No risk assessment data")
                                
                                # Clinical Summary
                                with ai_tab4:
                                    clinical_summary = ai_report.get("clinical_summary", "")
                                    if clinical_summary:
                                        st.info(clinical_summary)
                                    else:
                                        st.info("No summary available")
                                
                                # Full Report
                                with ai_tab5:
                                    full_report = ai_report.get("full_report", {})
                                    if full_report:
                                        st.json(full_report)
                                    else:
                                        st.info("No full report available")
                                
                                # Timestamp
                                st.caption(f"Generated: {alert['timestamp']}")
                        
                        # Save alerts to file
                        if st.button("Save Alerts to File", use_container_width=True):
                            alerts_file = f"doctor_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            alerts_data = {
                                "timestamp": datetime.now().isoformat(),
                                "total_alerts": len(st.session_state.alerts),
                                "alerts": st.session_state.alerts,
                                "ai_agent_reports": st.session_state.ai_agent_reports
                            }
                            with open(alerts_file, 'w', encoding='utf-8') as f:
                                json.dump(alerts_data, f, indent=2)
                            st.success(f"Alerts saved to {alerts_file}")
                    
                    # Export predictions
                    st.divider()
                    if st.button("Export Predictions as CSV"):
                        export_df = generated_df.copy()
                        export_df['prediction'] = st.session_state.predictions
                        export_df['heart_disease_risk_%'] = (st.session_state.probabilities[:, 1] * 100).round(2)
                        export_df['risk_level'] = export_df['heart_disease_risk_%'].apply(
                            lambda x: 'HIGH RISK' if x >= 70 else ('MEDIUM RISK' if x >= 40 else 'LOW RISK')
                        )
                        
                        csv_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        export_df.to_csv(csv_file, index=False)
                        st.success(f"Predictions exported to {csv_file}")
            
            # Auto-refresh handler (logic only - UI is at the top)
            if st.session_state.get('auto_refresh_enabled', False):
                
                # Auto-generate and predict every 5 seconds
                st.session_state.countdown = st.session_state.get('countdown', 5)
                
                if st.session_state.countdown > 0:
                    st.info(f"Next refresh in {st.session_state.countdown} seconds...")
                    st.session_state.countdown -= 1
                    time.sleep(1)
                    st.rerun()
                else:
                    # Generate new data
                    st.session_state.generated_data = generate_random_patient_data(num_patients, feature_names)
                    st.session_state.predictions = None
                    st.session_state.alerts = []
                    st.session_state.ai_agent_reports = {}
                    
                    # Make predictions automatically
                    try:
                        predictions, probabilities = predict_heart_disease(st.session_state.generated_data, model, scaler, feature_names)
                        st.session_state.predictions = predictions
                        st.session_state.probabilities = probabilities
                        
                        st.session_state.alerts = []
                        if 'notifications' not in st.session_state:
                            st.session_state.notifications = []
                        
                        for idx, (prob, pred) in enumerate(zip(probabilities, predictions)):
                            risk_pct = prob[1] * 100
                            patient_id = f"Patient_{idx+1}"
                            patient_dict = st.session_state.generated_data.iloc[idx].to_dict()
                            
                            if risk_pct >= 60:
                                alert = create_doctor_alert(risk_pct, patient_id, patient_dict)
                                st.session_state.alerts.append(alert)
                                
                                ai_analysis = process_patient_through_ai_agents(patient_dict, patient_id)
                                st.session_state.ai_agent_reports[patient_id] = ai_analysis
                                
                                notification = {
                                    "patient_id": patient_id,
                                    "risk_level": alert['risk_level'],
                                    "risk_percentage": alert['risk_percentage'],
                                    "timestamp": datetime.now().isoformat(),
                                    "vitals": alert['vitals'],
                                    "ai_report": ai_analysis
                                }
                                st.session_state.notifications.append(notification)
                    except Exception as e:
                        st.error(f"Error in auto-refresh: {str(e)}")
                    
                    # Reset countdown
                    st.session_state.countdown = 5
                    time.sleep(1)
                    st.rerun()

# Page: Run Analysis Tab
with main_tabs[2]:
    st.header("Run Patient Analysis")
    st.markdown("Upload dataset and generate health reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_path = st.text_input("Dataset path", value="dataset.csv")
        num_patients = st.slider("Number of patients to analyze", 1, 50, 5)
    
    with col2:
        use_ml = st.checkbox("Use ML models for summaries", value=False)
        st.caption("⚠️ ML models take longer but provide better summaries")
    
    if st.button("🚀 Run Analysis", key="run_analysis"):
        with st.spinner("Analyzing patient data..."):
            try:
                # Load data
                df = load_data(dataset_path)
                groups = get_patient_groups(df)
                
                # Initialize agents
                monitoring = MonitoringAgent()
                analysis = AnalysisAgent()
                anomaly_det = AnomalyDetectionAgent()
                risk_pred = RiskPredictionAgent()
                generative = GenerativeAgent(use_ml=use_ml)
                alert_report = AlertReportAgent()
                
                all_reports = []
                progress_bar = st.progress(0)
                
                for idx, (patient_id, patient_df) in enumerate(groups.items()):
                    if idx >= num_patients:
                        break
                    
                    # Convert to list of dicts
                    patient_data = patient_df.to_dict(orient="records")
                    if not patient_data:
                        continue
                    
                    latest = patient_data[-1]
                    
                    # Run all agents
                    alerts = monitoring.act(latest)
                    stats = analysis.act(patient_data)
                    anomalies = anomaly_det.act(latest, stats)
                    risk_data = risk_pred.act(latest, stats)
                    summary = generative.act(patient_id, latest, stats)
                    
                    report = alert_report.act(patient_id, alerts, anomalies, risk_data, summary)
                    all_reports.append(report)
                    
                    progress_bar.progress((idx + 1) / min(num_patients, len(groups)))
                
                # Save reports
                with open("patient_reports.json", "w", encoding='utf-8') as f:
                    json.dump(all_reports, f, indent=2)
                
                st.success(f"Analysis complete! Generated {len(all_reports)} patient reports.")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

# Page: View Reports Tab
with main_tabs[3]:
    st.header("Patient Reports")
    
    reports = load_reports()
    
    if not reports:
        st.info("📂 No reports available. Generate reports from the 'Run Analysis' page.")
    else:
        # Sort reports by severity: CRITICAL, HIGH, MEDIUM, LOW
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sorted_reports = sorted(
            reports, 
            key=lambda x: severity_order.get(x.get("severity"), 4)
        )
        
        st.subheader(f"Displaying {len(sorted_reports)} patient reports")
        
        for idx, report in enumerate(sorted_reports):
            with st.expander(f"📋 {report.get('patient_id')} - {report.get('severity')} Severity", expanded=(idx == 0)):
                display_patient_card(report)
                st.divider()
                display_alert_details(report)
                st.divider()
                display_risk_assessment(report)
                st.divider()
                display_clinical_summary(report)

# Page: Analytics Tab
with main_tabs[4]:
    st.header("Analytics & Visualizations")
    
    reports = load_reports()
    
    if not reports:
        st.warning("📊 No data available for analytics. Generate reports first.")
    else:
        # Convert reports to DataFrame for analysis
        df_reports = pd.json_normalize(reports)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity distribution
            severity_data = df_reports["severity"].value_counts().reset_index()
            severity_data.columns = ["severity", "count"]
            fig_severity = px.pie(
                severity_data,
                values="count",
                names="severity",
                title="Patient Distribution by Severity",
                color_discrete_map={
                    "CRITICAL": "#ff4444",
                    "HIGH": "#ff9800",
                    "MEDIUM": "#ffc107",
                    "LOW": "#4caf50"
                }
            )
            st.plotly_chart(fig_severity, width='stretch')
        
        with col2:
            # Risk score distribution - handle flattened column names from json_normalize
            risk_scores = []
            for report in reports:
                if isinstance(report.get("risk_assessment"), dict):
                    risk_scores.append(report["risk_assessment"].get("risk_score", 0))
            
            if risk_scores:
                fig_risk = px.histogram(
                    risk_scores,
                    nbins=10,
                    title="Risk Score Distribution",
                    labels={"value": "Risk Score", "count": "Number of Patients"}
                )
                st.plotly_chart(fig_risk, width='stretch')
            else:
                st.info("No risk score data available")
        
        # Alerts summary
        st.subheader("Alert Summary")
        alerts_data = []
        for report in reports:
            for alert in report.get("alerts", []):
                alerts_data.append({"Patient": report.get("patient_id"), "Alert": alert})
        
        if alerts_data:
            alerts_df = pd.DataFrame(alerts_data)
            alert_counts = alerts_df["Alert"].value_counts().reset_index()
            alert_counts.columns = ["alert", "count"]
            
            fig_alerts = px.bar(
                alert_counts,
                x="count",
                y="alert",
                orientation="h",
                title="Most Common Alerts",
                labels={"count": "Count", "alert": "Alert Type"}
            )
            st.plotly_chart(fig_alerts, width='stretch')
        else:
            st.info("No alerts recorded")
        
        # Detailed statistics
        st.subheader("Statistical Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_scores = []
            for report in reports:
                if isinstance(report.get("risk_assessment"), dict):
                    risk_scores.append(report["risk_assessment"].get("risk_score", 0))
            avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
            st.metric("Average Risk Score", f"{avg_risk:.1f}")
        
        with col2:
            total_alerts = sum(len(r.get("alerts", [])) for r in reports)
            st.metric("Total Alerts", total_alerts)
        
        with col3:
            critical_pct = (len([r for r in reports if r.get("severity") == "CRITICAL"]) / len(reports) * 100) if reports else 0
            st.metric("Critical Percentage", f"{critical_pct:.1f}%")
