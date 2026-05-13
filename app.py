from flask import Flask, render_template, request, make_response, redirect
import joblib
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch
from io import BytesIO
from datetime import datetime

app = Flask(__name__)# Health Progress Tracker - stores session history
prediction_history = []

model  = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    patient_name = request.form.get('patient_name', 'Unknown')
    features = [
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['bloodpressure']),
        float(request.form['skinthickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['dpf']),
        float(request.form['age'])
    ]
    data = scaler.transform([features])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    result = 'Diabetic' if prediction == 1 else 'Not Diabetic'
    confidence = round(probability * 100, 2)
    # Save to progress tracker
    prediction_history.append({
        'name': patient_name,
        'glucose': features[1],
        'bmi': features[5],
        'age': features[7],
        'result': result,
        'confidence': confidence,
        'date': datetime.now().strftime('%d %b %Y %I:%M %p')
    })
    # Keep only last 10 records
    if len(prediction_history) > 20:
        prediction_history.pop(0)

    return render_template('result.html',
                           result=result,
                           confidence=confidence,
                           features=features,
                           patient_name=patient_name)
@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        patient_name = request.form.get('patient_name', 'Unknown')
        result       = request.form.get('result', 'Unknown')
        confidence   = request.form.get('confidence', '0')
        features     = request.form.getlist('features')

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                rightMargin=50, leftMargin=50,
                                topMargin=40, bottomMargin=40)
        elements = []

        title_style = ParagraphStyle('title',
            fontSize=20, textColor=colors.HexColor('#062B5E'),
            spaceAfter=4, fontName='Helvetica-Bold', alignment=1)
        sub_style = ParagraphStyle('sub',
            fontSize=11, textColor=colors.HexColor('#667eea'),
            spaceAfter=3, alignment=1)
        date_style = ParagraphStyle('date',
            fontSize=10, textColor=colors.grey,
            spaceAfter=10, alignment=1)
        section_style = ParagraphStyle('section',
            fontSize=13, textColor=colors.HexColor('#062B5E'),
            spaceAfter=8, fontName='Helvetica-Bold')
        tip_style = ParagraphStyle('tip',
            fontSize=10, textColor=colors.HexColor('#444444'),
            spaceAfter=4, leftIndent=20)
        footer_style = ParagraphStyle('footer',
            fontSize=8, textColor=colors.grey,
            spaceAfter=2, alignment=1)

        elements.append(Paragraph("Diabetes Prediction Report", title_style))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph("AI-Powered Health Prediction System", sub_style))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(f"Patient Name: {patient_name}", sub_style))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", date_style))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#667eea')))
        elements.append(Spacer(1, 10))

        result_color = colors.HexColor('#e74c3c') if result == 'Diabetic' else colors.HexColor('#27ae60')
        result_style = ParagraphStyle('result',
            fontSize=18, textColor=result_color,
            spaceAfter=4, fontName='Helvetica-Bold', alignment=1)
        elements.append(Paragraph(f"Prediction Result: {result}", result_style))

        conf_style = ParagraphStyle('conf',
            fontSize=13, textColor=colors.HexColor('#4F46E5'),
            spaceAfter=4, alignment=1)
        elements.append(Paragraph(f"Confidence Level: {confidence}%", conf_style))

        conf_val = float(confidence)
        if conf_val < 40:
            risk = "LOW RISK"
            risk_color = colors.HexColor('#27ae60')
        elif conf_val < 70:
            risk = "MEDIUM RISK"
            risk_color = colors.HexColor('#f39c12')
        else:
            risk = "HIGH RISK"
            risk_color = colors.HexColor('#e74c3c')

        risk_style = ParagraphStyle('risk',
            fontSize=13, textColor=risk_color,
            spaceAfter=10, fontName='Helvetica-Bold', alignment=1)
        elements.append(Paragraph(f"Risk Level: {risk}", risk_style))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("Patient Health Parameters", section_style))

        labels = ['Pregnancies', 'Glucose Level', 'Blood Pressure',
                  'Skin Thickness', 'Insulin', 'BMI',
                  'Diabetes Pedigree Function', 'Age']
        units  = ['Count', 'mg/dL', 'mm Hg', 'mm', 'mu U/ml', 'kg/m2', 'Score', 'Years']

        table_data = [['Parameter', 'Value', 'Unit']]
        for i in range(len(labels)):
            val = features[i] if i < len(features) else 'N/A'
            table_data.append([labels[i], str(val), units[i]])

        table = Table(table_data, colWidths=[2.8*inch, 1.5*inch, 1.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor('#062B5E')),
            ('TEXTCOLOR',     (0,0), (-1,0),  colors.white),
            ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
            ('FONTSIZE',      (0,0), (-1,0),  11),
            ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.HexColor('#F0F4FF'), colors.white]),
            ('FONTSIZE',      (0,1), (-1,-1), 10),
            ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#e0e0e0')),
            ('TOPPADDING',    (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 10))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("Health Recommendations", section_style))

        if result == 'Diabetic':
            tips = [
                "Consult a doctor immediately for proper medical diagnosis.",
                "Avoid sugary foods. Eat more vegetables and lean proteins.",
                "Exercise at least 30 minutes daily.",
                "Drink at least 8 glasses of water daily.",
                "Monitor your blood glucose regularly.",
                "Get 7-8 hours of sleep every night.",
            ]
        else:
            tips = [
                "Maintain a balanced diet rich in fruits and vegetables.",
                "Exercise regularly — at least 150 minutes per week.",
                "Maintain a healthy body weight.",
                "Avoid smoking and limit alcohol consumption.",
                "Get regular health checkups every 6 months.",
                "Manage stress through meditation or yoga.",
            ]

        for i, tip in enumerate(tips):
            elements.append(Paragraph(f"{i+1}. {tip}", tip_style))

        elements.append(Spacer(1, 8))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#667eea')))
        elements.append(Spacer(1, 6))

        elements.append(Paragraph("This report is generated by Diabetes Prediction System - ML Project", footer_style))
        elements.append(Paragraph("Developed by Chandra Mowliswaran P | Machine Learning with Python Internship 2026", footer_style))
        elements.append(Paragraph("This is not a medical diagnosis. Please consult a qualified doctor.", footer_style))

        doc.build(elements)
        buffer.seek(0)

        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=Diabetes_Report.pdf'
        return response

    except Exception as e:
        return f"PDF Error: {str(e)}", 500
@app.route('/tracker')
def tracker():
    return render_template('tracker.html',
                           history=prediction_history)
@app.route('/clear_history')
def clear_history():
    prediction_history.clear()
    return redirect('/tracker')

if __name__ == '__main__':
    app.run(debug=True)