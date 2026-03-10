from flask import Flask, render_template, Response, request, send_from_directory, redirect, url_for, session # type: ignore
from sqlalchemy import func, case
from flask import jsonify
import cv2
import os
import time
import easyocr
import threading
from ultralytics import YOLO
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import pagesizes
from io import BytesIO
from flask import send_file

import re


def clean_plate_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)

    pattern = r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}'
    match = re.search(pattern, text)

    if match:
        return match.group()
    else:
        return text
from flask_sqlalchemy import SQLAlchemy # type: ignore

# -----------------------------
# APP CONFIG
# -----------------------------
app = Flask(__name__)
app.secret_key = "helmet_secret_key_2026"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///violations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

CAMERA_SOURCES = {
    "webcam": 0,
    "mobile": "http://192.168.2.105:8080/video"
}

current_camera = "webcam"

def open_camera():

    source = CAMERA_SOURCES[current_camera]

    if source == 0:
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    else:
        cam = cv2.VideoCapture(source)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cam.isOpened():
        print("Camera failed to open:", source)

    return cam

camera = open_camera()

# -----------------------------
# DATABASE MODEL
# -----------------------------
class Violation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(20))
    time = db.Column(db.String(20))
    image_name = db.Column(db.String(200))
    confidence = db.Column(db.Float)
    plate_number = db.Column(db.String(20))
    plate_confidence = db.Column(db.Float)
    plate_image = db.Column(db.String(200))
    helmet_type = db.Column(db.String(20))

with app.app_context():
    db.create_all()

# -----------------------------
# ADMIN LOGIN
# -----------------------------
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "1234"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO("../runs/detect/train2/weights/best.pt")
plate_model = YOLO("plate_model.pt")

reader = easyocr.Reader(['en'], gpu=False)

# -----------------------------
# FOLDERS
# -----------------------------
VIOLATION_FOLDER = "static/violations"
os.makedirs(VIOLATION_FOLDER, exist_ok=True)

PLATE_FOLDER = "static/plates"
os.makedirs(PLATE_FOLDER, exist_ok=True)

current_frame = None
output_frame = None


# -----------------------------
# SAVE TO DATABASE
# -----------------------------
def save_violation(filename, confidence, plate_text, plate_conf, plate_image, helmet_type):

    with app.app_context():

        now = datetime.now()

        violation = Violation(
            date=now.strftime("%d-%m-%Y"),
            time=now.strftime("%H:%M:%S"),
            image_name=filename,
            confidence=round(confidence, 2),
            plate_number=plate_text,
            plate_confidence=round(plate_conf, 2),
            plate_image=plate_image,
            helmet_type=helmet_type
        )

        db.session.add(violation)
        db.session.commit()

        print("Saved to database:", filename)

#background detection 

def detection_loop():

    global current_frame, output_frame

    processed_ids = set()

    while True:

        if current_frame is None:
            time.sleep(0.05)
            continue

        frame = current_frame.copy()

        try:

            results = model.track(frame, persist=True, imgsz=192, conf=0.5)

            for result in results:

                boxes = result.boxes

                if boxes is None or boxes.id is None:
                    continue

                for box, track_id in zip(boxes, boxes.id):

                    track_id = int(track_id)

                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]

                    if label not in ["with helmet", "without helmet"]:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if label == "without helmet":
                        color = (0,0,255)
                    else:
                        color = (0,255,0)

                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

                    cv2.putText(
                        frame,
                        f"{label} ID:{track_id}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                    # Save violation once
                    if label == "without helmet" and track_id not in processed_ids:

                        filename = f"violation_ID{track_id}_{int(time.time())}.jpg"
                        filepath = os.path.join(VIOLATION_FOLDER, filename)

                        cv2.imwrite(filepath, frame)

                        save_violation(
                            filename,
                            conf,
                            "Not Detected",
                            0,
                            None,
                            label
                        )

                        processed_ids.add(track_id)

        except Exception as e:
            print("Detection Error:", e)

        # Update shared frame for streaming
        output_frame = frame

        time.sleep(0.02)

# -----------------------------
# VIDEO STREAM
# -----------------------------
#camera = cv2.VideoCapture(0)
def generate_frames():

    global camera, current_frame, output_frame

    while True:

        if not camera.isOpened():
            print("Camera reconnecting...")
            camera.release()
            camera = open_camera()
            time.sleep(1)
            continue

        success, frame = camera.read()

        if not success or frame is None:
            print("Frame read failed... reconnecting camera")
            camera.release()
            time.sleep(1)
            camera = open_camera()
            continue

        # resize for faster processing
        frame = cv2.resize(frame, (320,240))
        frame = cv2.flip(frame,1)

        # send frame to detection thread
        current_frame = frame.copy()

        # if detection hasn't processed yet, show raw frame
        display_frame = output_frame if output_frame is not None else frame

        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.03)
         


# -----------------------------
# ROUTES
# -----------------------------
@app.route('/')
def index():
    return render_template("index.html")
# -----------------------------
# video
# -----------------------------

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------
# LOGIN
# -----------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin'] = True
            return redirect(url_for('gallery'))
        else:
            return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")

@app.route('/logout')
def logout():
    session.pop('admin', None)
    return redirect(url_for('login'))


#
# camera switch
#
@app.route('/switch_camera/<cam>')
def switch_camera(cam):

    global camera, current_camera

    if cam in CAMERA_SOURCES:

        camera.release()

        current_camera = cam
        camera = open_camera()

        print("Camera switched to:", cam)

    return redirect(url_for('index'))

# -----------------------------
# GALLERY
# -----------------------------
@app.route('/gallery')
def gallery():

    if not session.get('admin'):
        return redirect(url_for('login'))

    violations = Violation.query.order_by(Violation.id.desc()).all()

    valid_violations = []

    for v in violations:

        if not v.image_name:
            continue

        image_path = os.path.join(VIOLATION_FOLDER, v.image_name)

        if os.path.exists(image_path):
            valid_violations.append(v)

    image_data = [{
        "name": v.image_name,
        "date": v.date,
        "time": v.time,
        "confidence": v.confidence
    } for v in valid_violations]

    return render_template(
        "gallery.html",
        images=image_data,
        count=len(image_data)
    )

# -----------------------------
# DELETE
# -----------------------------
@app.route('/delete/<filename>')
def delete(filename):

    if not session.get('admin'):
        return redirect(url_for('login'))

    violation = Violation.query.filter_by(image_name=filename).first()

    if violation:
        db.session.delete(violation)
        db.session.commit()

    path = os.path.join(VIOLATION_FOLDER, filename)

    if os.path.exists(path):
        os.remove(path)

    return redirect(url_for('gallery'))
# -----------------------------
# DOWNLOAD
# -----------------------------
@app.route('/download/<filename>')
def download(filename):
    if not session.get('admin'):
        return redirect(url_for('login'))

    return send_from_directory(VIOLATION_FOLDER, filename, as_attachment=True)

# -----------------------------
# LOGS PAGE
# -----------------------------
@app.route('/logs', methods=['GET'])
def logs():
    if not session.get('admin'):
        return redirect(url_for('login'))

    selected_date = request.args.get('date')

    violations = Violation.query.order_by(
        Violation.date.desc(),
        Violation.time.desc()
    ).all()

    if selected_date:
        try:
            date_obj = datetime.strptime(selected_date, "%Y-%m-%d")
            converted_date = date_obj.strftime("%d-%m-%Y")
        except:
            converted_date = selected_date

        violations = [
            v for v in violations
            if v.date == converted_date
        ]

    # 🔥 Group by date
    grouped_data = {}
    for v in violations:
        grouped_data.setdefault(v.date, []).append(v)

    return render_template(
        "logs.html",
        grouped_data=grouped_data,
        selected_date=selected_date
    )
# chart
# -----------------------------
@app.route('/chart-data')
def chart_data():

    data = db.session.query(
        Violation.date,
        func.sum(
            case(
                (Violation.helmet_type == "with helmet", 1),
                else_=0
            )
        ).label("with_count"),
        func.sum(
            case(
                (Violation.helmet_type == "without helmet", 1),
                else_=0
            )
        ).label("without_count")
    ).group_by(Violation.date).order_by(Violation.date).all()

    if not data:
        return jsonify({
            "dates": [],
            "with_helmet": [],
            "without_helmet": []
        })

    return jsonify({
        "dates": [row[0] for row in data],
        "with_helmet": [row[1] for row in data],
        "without_helmet": [row[2] for row in data]
    })
#live count
#-----------------------------
@app.route('/live-count')
def live_count():

    total = Violation.query.count()

    with_helmet = Violation.query.filter_by(
        helmet_type="with helmet"
    ).count()

    without_helmet = Violation.query.filter_by(
        helmet_type="without helmet"
    ).count()

    compliance = 0
    if total > 0:
        compliance = round((with_helmet / total) * 100, 2)

    return jsonify({
        "total": total,
        "with_helmet": with_helmet,
        "without_helmet": without_helmet,
        "compliance": compliance
    })

# UPLOAD IMAGE
# -----------------------------
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']

        save_path = os.path.join("static", file.filename)
        file.save(save_path)

        img = cv2.imread(save_path)
        results = model(img)

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = (0, 0, 255) if label == "without helmet" else (0, 255, 0)

                text = f"{label} ({conf:.2f})"

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        result_image = f"result_{int(time.time())}.jpg"
        result_path = os.path.join("static", result_image)

        cv2.imwrite(result_path, img)

        return render_template("upload.html", result=result_image)

    return render_template("upload.html")


# download reports
#-----------------------------
@app.route('/download-report')
def download_report():

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=pagesizes.A4)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("Helmet Detection Analytics Report", styles['Title']))
    elements.append(Spacer(1, 20))

    data = [["Date", "With Helmet", "Without Helmet", "Total"]]

    stats = db.session.query(
        Violation.date,
        func.sum(
            case(
                (Violation.helmet_type == "with helmet", 1),
                else_=0
            )
        ),
        func.sum(
            case(
                (Violation.helmet_type == "without helmet", 1),
                else_=0
            )
        )
    ).group_by(Violation.date).order_by(Violation.date).all()

    for row in stats:
        date = row[0]
        with_count = row[1] or 0
        without_count = row[2] or 0
        total = with_count + without_count

        data.append([date, str(with_count), str(without_count), str(total)])

    table = Table(data)
    table.setStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('ALIGN', (1,1), (-1,-1), 'CENTER')
    ])

    elements.append(table)
    doc.build(elements)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="helmet_detection_report.pdf",
        mimetype='application/pdf'
    )

# -----------------------------
# MAIN
# -----------------------------

thread = threading.Thread(target=detection_loop, daemon=True)
thread.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)