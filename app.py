import os
import numpy as np
import librosa
import tensorflow as tf
import sqlite3
import json
import uuid
import secrets
from datetime import datetime
from io import BytesIO

# Imports required for Flask
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_file,
    jsonify,
    send_from_directory
)
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# Imports required for PDF generation with ReportLab
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from functools import wraps
from flask import abort, redirect, url_for, flash

# CORRECTED: The admin_required decorator now checks the 'role' attribute directly.
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        if current_user.role != 'admin':
            flash("You do not have permission to view this page.", "danger")
            return redirect(url_for('user_dashboard'))
        return f(*args, **kwargs)
    return decorated_function
# -------------------------------
# Suppress TensorFlow warnings
# -------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------------------
# Flask Setup
# -------------------------------
app = Flask(__name__)
app.secret_key = "secret123"
UPLOAD_FOLDER = "dataset/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "info"

# -------------------------------
# Database Setup
# -------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_FILE = os.path.join(BASE_DIR, "database", "database.db")
os.makedirs("database", exist_ok=True)

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

# UPDATED: Reusable DB connection function
def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            file_path TEXT NOT NULL,
            result TEXT,
            confidence REAL,
            admin_override TEXT,
            user_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    admin_user = conn.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()

    if not admin_user:
        admin_password = "admin123"
        hashed_password = generate_password_hash(admin_password, method='pbkdf2:sha256')
        cursor.execute(
            "INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
            ('admin', 'admin@example.com', hashed_password, 'admin')
        )
        print("✅ Default admin user created.")

    conn.commit()
    conn.close()

# -------------------------------
# User Model for Flask-Login
# -------------------------------
class User(UserMixin):
    def __init__(self, id, username, email, password, status, role):
        self.id = id
        self.username = username
        self.email = email
        self.password = password
        self.status = status
        self.role = role

    def get_id(self):
        return str(self.id)

    def is_active(self):
        return self.status == 'active'

    # ADDED: A property to check if the user is an admin
    @property
    def is_admin(self):
        return self.role == 'admin'

# -------------------------------
# User Loader
# -------------------------------
@login_manager.user_loader
def load_user(user_id):
    conn = get_db()
    user_data = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if user_data:
        return User(
            id=user_data['id'],
            username=user_data['username'],
            email=user_data['email'],
            password=user_data['password'],
            status=user_data['status'],
            role=user_data['role']
        )
    return None

# -------------------------------
# ML Model Setup (unmodified)
# -------------------------------
MODEL_PATH = "ml_models/final_cnn_bilstm.h5"
CLASSES = ["healthy_voices", "diseased_voices"]
SAMPLE_RATE = 16000
MAX_LEN = 160
N_MELS = 64
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
FEATURE_MEAN = np.load("ml_models/feature_mean.npy")
FEATURE_STD = np.load("ml_models/feature_std.npy")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y)
    y = y / (np.max(np.abs(y)) + 1e-9)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=256, n_fft=512)
    mel_db = librosa.power_to_db(mel, ref=np.max).T
    if mel_db.shape[0] < MAX_LEN:
        pad_width = MAX_LEN - mel_db.shape[0]
        mel_db = np.pad(mel_db, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mel_db = mel_db[:MAX_LEN, :]
    return mel_db

def predict_voice(file_path):
    try:
        feat = extract_features(file_path)
        feat = np.expand_dims(feat, axis=0)
        feat = (feat - FEATURE_MEAN) / (FEATURE_STD + 1e-9)
        pred = model.predict(feat)[0]
        class_idx = int(np.argmax(pred))
        confidence = float(pred[class_idx])
        result_class = CLASSES[class_idx].replace('_voices', '').capitalize()
        return result_class, confidence
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return "Error!", 0.0

# ADDED: New function to generate the PDF report, which is reusable
def generate_report_pdf(submission_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        alignment=TA_CENTER,
        spaceAfter=30
    )
    heading_style = ParagraphStyle(
        name='HeadingStyle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#3498db'),
        spaceBefore=20,
        spaceAfter=10
    )
    subheading_style = ParagraphStyle(
        name='SubheadingStyle',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceBefore=15,
        spaceAfter=8
    )
    normal_style = ParagraphStyle(
        name='NormalStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=6
    )
    result_style = ParagraphStyle(
        name='ResultStyle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.white,
        alignment=TA_CENTER
    )
    disclaimer_style = ParagraphStyle(
        name='DisclaimerStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#7f8c8d'),
        fontStyle='italic',
        spaceBefore=20
    )

    story = []
    header_data = [
        [
            Paragraph("<b>Voice Health Analysis Report</b>", title_style),
            Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y')}", normal_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[4 * inch, 2 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (0, 0), 'CENTER'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
    ]))
    story.append(header_table)

    story.append(Paragraph("User Information", heading_style))
    user_data = [
        ['Username:', submission_data['username']],
        ['Submission ID:', str(submission_data['id'])],
        ['Date of Test:', submission_data['created_at']]
    ]
    if 'admin_override' in submission_data and submission_data['admin_override']:
        user_data.append(['Admin Override:', submission_data['admin_override']])

    user_table = Table(user_data, colWidths=[1.5 * inch, 3 * inch])
    user_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e9ecef'))
    ]))
    story.append(user_table)

    story.append(Spacer(1, 20))
    story.append(Paragraph("Prediction Results", heading_style))
    result_color = colors.HexColor('#2ecc71') if submission_data['result'] == 'Healthy' else colors.HexColor('#e74c3c')
    result_text = f"<b>{submission_data['result']}</b>"
    result_data = [
        ['Result:', Paragraph(result_text, result_style)],
        ['Confidence:', f"{submission_data['confidence']:.2f}%"]
    ]
    result_table = Table(result_data, colWidths=[1.5 * inch, 3 * inch])
    result_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, 0), 'CENTER'),
        ('ALIGN', (1, 1), (1, 1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, 0), result_color),
        ('TEXTCOLOR', (1, 0), (1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e9ecef')),
        ('ROWBACKGROUNDS', (0, 1), (0, 1), [colors.HexColor('#f8f9fa')])
    ]))
    story.append(result_table)

    if submission_data['user_notes']:
        story.append(Spacer(1, 20))
        story.append(Paragraph("User Notes", subheading_style))
        story.append(Paragraph(submission_data['user_notes'], normal_style))

    story.append(Spacer(1, 20))
    story.append(Paragraph("Health Recommendations", subheading_style))
    if submission_data['result'] == 'Healthy':
        recommendations = [
            "Continue to maintain good vocal hygiene by staying hydrated.",
            "Avoid straining your voice and practice proper breathing techniques.",
            "Get adequate rest and maintain a healthy lifestyle.",
            "Consider regular voice check-ups if you use your voice professionally."
        ]
    else:
        recommendations = [
            "Consult with an ENT specialist or speech-language pathologist.",
            "Rest your voice as much as possible and avoid whispering.",
            "Stay hydrated by drinking plenty of water.",
            "Avoid irritants such as smoking, alcohol, and caffeine.",
            "Use a humidifier to maintain optimal humidity levels."
        ]
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", normal_style))

    story.append(Spacer(1, 30))
    disclaimer_text = """
    <b>Disclaimer:</b> This report is generated by an AI system and is intended for informational purposes only.
    It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of
    your physician or other qualified health provider with any questions you may have regarding a medical condition.
    Do not disregard professional medical advice or delay in seeking it because of something you have read in this report.
    """
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    def add_footer(canvas, doc):
        canvas.saveState()
        footer_text = f"Voice AI Health Report - Page {canvas.getPageNumber()}"
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.HexColor('#7f8c8d'))
        canvas.drawCentredString(4.25 * inch, 0.5 * inch, footer_text)
        canvas.restoreState()

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    buffer.seek(0)
    return buffer

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('user_dashboard'))
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('user_dashboard'))
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if user:
            flash("❌ Username already exists")
            conn.close()
            return redirect(url_for('register'))
        try:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            conn.execute("INSERT INTO users(username, email, password) VALUES(?,?,?)", (username, email, hashed_password))
            conn.commit()
            flash("✅ Registered successfully! Please login.")
            return redirect(url_for('login'))
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('user_dashboard'))

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = get_db()
        user_data = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        conn.close()

        if user_data and check_password_hash(user_data['password'], password):
            user = load_user(user_data['id'])
            if user and not user.is_active():
                flash("❌ Your account is blocked. Contact admin.")
                return redirect(url_for('login'))

            login_user(user)
            flash(f"✅ Welcome back, {user.username}!")
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('user_dashboard'))

        flash("❌ Invalid credentials")
    return render_template("login.html")
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("✅ You have been logged out.")
    return redirect(url_for('index'))

@app.route('/user_dashboard')
@login_required
def user_dashboard():
    conn = get_db()
    submissions_data = conn.execute(
        "SELECT * FROM submissions WHERE user_id=? ORDER BY created_at DESC",
        (current_user.id,)
    ).fetchall()
    conn.close()

    total_submissions = len(submissions_data)
    healthy_count = sum(1 for sub in submissions_data if sub['result'] == 'Healthy')
    diseased_count = total_submissions - healthy_count
    last_submission = submissions_data[0] if submissions_data else None

    return render_template(
        "user_dashboard.html",
        submissions=submissions_data,
        total_submissions=total_submissions,
        healthy_count=healthy_count,
        diseased_count=diseased_count,
        last_submission=last_submission
    )

@app.route('/user_predict_voice', methods=['POST'])
@login_required
def user_predict_voice():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.wav'):
        filename = f"{uuid.uuid4().hex}.wav"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        result, confidence = predict_voice(filepath)

        conn = get_db()
        conn.execute(
            "INSERT INTO submissions (user_id, file_path, result, confidence) VALUES (?, ?, ?, ?)",
            (current_user.id, filepath, result, confidence)
        )
        conn.commit()
        last_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        return jsonify({
            "result_class": result,
            "confidence": "%.2f" % (confidence * 100),
            "submission_id": last_id
        })
    return jsonify({"error": "Invalid file format"}), 400

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash("Access denied! Admins only.")
        return redirect(url_for('login'))

    conn = get_db()
    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    healthy_count = conn.execute("SELECT COUNT(*) FROM submissions WHERE result='Healthy'").fetchone()[0]
    diseased_count = conn.execute("SELECT COUNT(*) FROM submissions WHERE result='Diseased'").fetchone()[0]
    users = conn.execute("SELECT * FROM users").fetchall()
    submissions = conn.execute("""
        SELECT s.id, u.username, s.file_path, s.result, s.confidence, s.admin_override, s.created_at
        FROM submissions s LEFT JOIN users u ON s.user_id = u.id ORDER BY s.created_at DESC
    """).fetchall()
    conn.close()

    metrics = None
    metrics_path = "ml_models/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return render_template(
        "admin_dashboard.html",
        users=users,
        submissions=submissions,
        classes=CLASSES,
        metrics=metrics,
        total_users=total_users,
        healthy_count=healthy_count,
        diseased_count=diseased_count,
    )

@app.route("/admin/override/<int:sub_id>", methods=["POST"])
@login_required
def admin_override(sub_id):
    if current_user.role != "admin":
        return redirect(url_for('login'))
    override = request.form["override"]
    conn = get_db()
    conn.execute("UPDATE submissions SET admin_override=? WHERE id=?", (override, sub_id))
    conn.commit()
    conn.close()
    flash(f"⚡ Submission {sub_id} manually overridden to {override}")
    return redirect(url_for('admin_dashboard'))

@app.route("/admin_upload_data", methods=["POST"])
@login_required
def admin_upload_data():
    if current_user.role != "admin":
        return redirect(url_for('login'))
    file = request.files.get("file")
    category = request.form.get("category")
    if not file or category not in CLASSES:
        flash("❌ Invalid file or category")
        return redirect(url_for('admin_dashboard'))
    save_dir = os.path.join(UPLOAD_FOLDER, category)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, secure_filename(file.filename))
    file.save(path)
    flash(f"📥 Training data uploaded: {file.filename}")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin_test_voice', methods=['POST'])
@login_required
def admin_test_voice():
    if current_user.role != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    file_path = None
    try:
        if 'file' in request.files:
            file = request.files['file']
            original_filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, original_filename)
            file.save(file_path)

            file_to_predict = file_path
        else:
            return jsonify({"error": "No audio data received"}), 400

        result, confidence = predict_voice(file_to_predict)

        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({
            "result_class": result,
            "confidence": round(float(confidence) * 100, 2)
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500

@app.route("/admin/block/<int:user_id>", methods=["POST"])
@login_required
def block_user(user_id):
    if current_user.role != "admin":
        return redirect(url_for('login'))
    conn = get_db()
    conn.execute("UPDATE users SET status='blocked' WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    flash("User blocked successfully.")
    return redirect(url_for('admin_dashboard'))

@app.route("/admin/unblock/<int:user_id>", methods=["POST"])
@login_required
def unblock_user(user_id):
    if current_user.role != "admin":
        return redirect(url_for('login'))
    conn = get_db()
    conn.execute("UPDATE users SET status='active' WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    flash("User unblocked successfully.")
    return redirect(url_for('admin_dashboard'))

@app.route("/admin/delete/<int:user_id>", methods=["POST"])
@login_required
def delete_user(user_id):
    if current_user.role != "admin":
        return redirect(url_for('login'))
    conn = get_db()
    conn.execute("DELETE FROM submissions WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    flash("User and all their data deleted successfully.")
    return redirect(url_for('admin_dashboard'))

# Modifed to return a JSON response
@app.route('/user/delete_submission/<int:sub_id>', methods=['POST'])
@login_required
def delete_user_submission(sub_id):
    conn = get_db()
    submission = conn.execute("SELECT * FROM submissions WHERE id=? AND user_id=?", (sub_id, current_user.id)).fetchone()
    if not submission:
        conn.close()
        return jsonify({"success": False, "message": "Submission not found or you are not authorized to delete it."}), 404
    file_path = submission['file_path']
    conn.execute("DELETE FROM submissions WHERE id=?", (sub_id,))
    conn.commit()
    conn.close()
    if os.path.exists(file_path):
        os.remove(file_path)
    return jsonify({"success": True, "message": "Submission deleted successfully.", "sub_id": sub_id})

@app.route('/admin/delete_submission/<int:sub_id>', methods=['POST'])
@login_required
def delete_admin_submission(sub_id):
    if current_user.role != "admin":
        return redirect(url_for('login'))
    conn = get_db()
    submission = conn.execute("SELECT * FROM submissions WHERE id=?", (sub_id,)).fetchone()
    if not submission:
        flash("❌ Submission not found.")
        conn.close()
        return redirect(url_for('admin_dashboard'))
    file_path = submission['file_path']
    conn.execute("DELETE FROM submissions WHERE id=?", (sub_id,))
    conn.commit()
    conn.close()
    if os.path.exists(file_path):
        os.remove(file_path)
    flash("✅ Submission deleted successfully.")
    return redirect(url_for('admin_dashboard'))

@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/download_user_report/<int:sub_id>')
@login_required
def download_user_report(sub_id):
    conn = get_db()
    submission = conn.execute("""
        SELECT s.id, u.username, s.file_path, s.result, s.confidence, s.created_at, s.user_notes
        FROM submissions s LEFT JOIN users u ON s.user_id = u.id
        WHERE s.id = ? AND s.user_id = ?
    """, (sub_id, current_user.id)).fetchone()
    conn.close()

    if not submission:
        flash("Report not found or not authorized.")
        return redirect(url_for('user_dashboard'))

    try:
        buffer = generate_report_pdf(submission)
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'voice_health_report_{submission["id"]}.pdf'
        )
    except Exception as e:
        flash(f"Error generating report: {e}", "danger")
        return redirect(url_for('user_dashboard'))

# DELETED: This route is now redundant.
# @app.route('/download_admin_report/<int:sub_id>')

# Modified to return a JSON response
@app.route('/user/delete_notes/<int:sub_id>', methods=['POST'])
@login_required
def delete_user_notes(sub_id):
    conn = get_db()
    submission = conn.execute(
        "SELECT * FROM submissions WHERE id=? AND user_id=?",
        (sub_id, current_user.id)
    ).fetchone()
    if not submission:
        conn.close()
        return jsonify({"success": False, "message": "Submission not found or you are not authorized."}), 404

    conn.execute("UPDATE submissions SET user_notes=NULL WHERE id=?", (sub_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Notes deleted successfully!"})

# Modified to return a JSON response
@app.route('/user/update_notes/<int:sub_id>', methods=['POST'])
@login_required
def update_user_notes(sub_id):
    conn = get_db()
    notes = request.form.get('notes')
    submission = conn.execute(
        "SELECT * FROM submissions WHERE id=? AND user_id=?",
        (sub_id, current_user.id)
    ).fetchone()
    if not submission:
        conn.close()
        return jsonify({"success": False, "message": "Submission not found or you are not authorized."}), 404

    conn.execute("UPDATE submissions SET user_notes=? WHERE id=? AND user_id=?", (notes, sub_id, current_user.id))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "Notes updated successfully!", "notes": notes})

# New route for Admin to download any user's report
@app.route('/admin/download_report/<int:sub_id>')
@login_required
@admin_required
def admin_download_report(sub_id):
    conn = get_db_connection()
    submission = conn.execute("""
        SELECT s.id, u.username, s.file_path, s.result, s.confidence, s.created_at, s.admin_override, s.user_notes
        FROM submissions s LEFT JOIN users u ON s.user_id = u.id
        WHERE s.id = ?
    """, (sub_id,)).fetchone()
    conn.close()

    if not submission:
        flash("Submission not found.", "danger")
        return redirect(url_for('admin_dashboard'))

    try:
        buffer = generate_report_pdf(submission)
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'voice_health_report_{submission["id"]}.pdf'
        )
    except Exception as e:
        flash(f"Error generating report: {e}", "danger")
        return redirect(url_for('admin_dashboard'))

if __name__ == "__main__":
    init_db()
    app.run(debug=True)