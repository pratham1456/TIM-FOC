import time
import os
import cv2
import threading
import neoapi
from flask import Flask, request, jsonify, render_template, Response, send_file
from grayscale import convert_to_grayscale
from client_peel_off import peel_off
from client_material_defect import material_defect
from client_plating_defect import plating_defect
from client_dent import dent

app = Flask(__name__)
USE_BAUMER = True   # Always use Baumer camera

# Shared frame buffer and lock
global_frame = None
frame_lock = threading.Lock()

# Directories for saving images
captured_dir = 'static/capturedImage'
detected_dir = 'static/detectedImage'
os.makedirs(captured_dir, exist_ok=True)
os.makedirs(detected_dir, exist_ok=True)


def camera_grabber():
    """Continuously capture frames from the Baumer camera into a shared buffer."""
    global global_frame
    try:
        cam = neoapi.Cam()
        cam.Connect('700009367129')
        cam.f.ExposureTime.Set(400)
    except Exception as e:
        print(f"Error connecting to Baumer camera in grabber: {e}")
        return

    while True:
        try:
            img = cam.GetImage()
            if img.IsEmpty():
                time.sleep(0.01)
                continue
            f = img.GetNPArray()
        except Exception as e:
            print(f"Baumer frame grab error: {e}")
            time.sleep(0.05)
            continue

        # Update shared buffer
        with frame_lock:
            global_frame = f.copy()

        # Small sleep to limit frame rate
        time.sleep(0.01)


def generate_frames():
    """Yield the latest frame as multipart JPEG for MJPEG streaming."""
    global global_frame
    while True:
        with frame_lock:
            f = global_frame.copy() if global_frame is not None else None

        if f is None:
            time.sleep(0.01)
            continue

        _, buf = cv2.imencode('.jpg', f)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/capture', methods=['POST'])
def capture_image():
    """Capture the current shared frame, process, save, and return URLs."""
    global global_frame
    # Grab the latest frame
    with frame_lock:
        f = global_frame.copy() if global_frame is not None else None

    if f is None:
        return jsonify({'error': 'No frame available'}), 500

    # Overlay grid if needed (already applied in grabber)
    # f = draw_grid(f)

    # Save the raw capture
    cap_path = os.path.join(captured_dir, 'captured.jpg')
    cv2.imwrite(cap_path, f)

    # Process based on posted button
    data = request.get_json() or {}
    button = int(data.get('button', 0))
    if button == 1:
        out_img,msg = convert_to_grayscale(cap_path, detected_dir)
    elif button == 2:
        out_img,msg = peel_off(cap_path)
    elif button == 3:
        out_img,msg = material_defect(cap_path)
    elif button == 4:
        out_img,msg = plating_defect(cap_path)
    elif button == 5:
        out_img,msg = dent(cap_path)
    else:
         msg = f, 'No processing selected.'

    # Save detected image
    det_path = os.path.join(detected_dir, 'detected.jpg')
    new_width = int(out_img.shape[1] * 0.7)
    new_height = int(out_img.shape[0] * 0.7)
    resized_op_img=cv2.resize(out_img,(new_width,new_height))
    cv2.imwrite(det_path, resized_op_img)

    return jsonify({
        'captured_image': '/static/capturedImage/captured.jpg',
        'detected_image': '/static/detectedImage/detected.jpg',
        'message': msg
    })


@app.route('/get_captured_image')
def get_captured_image():
    return send_file(os.path.join(captured_dir, 'captured.jpg'))


@app.route('/get_detected_image')
def get_detected_image():
    return send_file(os.path.join(detected_dir, 'detected.jpg'))


if __name__ == '__main__':
    # Start the camera grabber thread (daemon so it stops with main)
    t = threading.Thread(target=camera_grabber, daemon=True)
    t.start()

    # Run Flask without reloader to avoid double threads
    app.run(debug=True, use_reloader=False)
