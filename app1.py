import time
import os
import cv2
import threading
from flask import Flask, request, jsonify, render_template, Response, send_file
from grayscale import convert_to_grayscale
from client_peel_off import peel_off
from client_material_defect import material_defect
from client_plating_defect import plating_defect
from client_dent import dent
import depthai as dai

app = Flask(__name__)
USE_BAUMER = False  # Always use Baumer camera

# Shared frame buffer and lock
global_frame = None
frame_lock = threading.Lock()

# Directories for saving images
captured_dir = "static/capturedImage"
detected_dir = "static/detectedImage"
os.makedirs(captured_dir, exist_ok=True)
os.makedirs(detected_dir, exist_ok=True)


def generate_frames(buffer_name):
    global global_frame_cam1, global_frame_cam2
    while True:
        with frame_lock:
            if buffer_name == "cam1":
                f = global_frame_cam1.copy() if global_frame_cam1 is not None else None
            else:
                f = global_frame_cam2.copy() if global_frame_cam2 is not None else None

        if f is None:
            time.sleep(0.01)
            continue

        _, buf = cv2.imencode(".jpg", f)
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/video_feed_cam1")
def video_feed_cam1():
    return Response(
        generate_frames("cam1"), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/video_feed_cam2")
def video_feed_cam2():
    return Response(
        generate_frames("cam2"), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/capture", methods=["POST"])
def capture_image():
    """Capture the current shared frame, process, save, and return URLs."""
    global global_frame
    # Grab the latest frame
    with frame_lock:
        f = global_frame.copy() if global_frame is not None else None

    if f is None:
        return jsonify({"error": "No frame available"}), 500

    # Overlay grid if needed (already applied in grabber)
    # f = draw_grid(f)

    # Save the raw capture
    cap_path = os.path.join(captured_dir, "captured.jpg")
    cv2.imwrite(cap_path, f)

    # Process based on posted button
    data = request.get_json() or {}
    button = int(data.get("button", 0))
    if button == 1:
        out_img, msg = convert_to_grayscale(cap_path, detected_dir)
    elif button == 2:
        out_img, msg = peel_off(cap_path)
    elif button == 3:
        out_img, msg = material_defect(cap_path)
    elif button == 4:
        out_img, msg = plating_defect(cap_path)
    elif button == 5:
        out_img, msg = dent(cap_path)
    else:
        msg = f, "No processing selected."

    # Save detected image
    det_path = os.path.join(detected_dir, "detected.jpg")
    new_width = int(out_img.shape[1] * 0.7)
    new_height = int(out_img.shape[0] * 0.7)
    resized_op_img = cv2.resize(out_img, (new_width, new_height))
    cv2.imwrite(det_path, resized_op_img)

    return jsonify(
        {
            "captured_image": "/static/capturedImage/captured.jpg",
            "detected_image": "/static/detectedImage/detected.jpg",
            "message": msg,
        }
    )


@app.route("/get_captured_image")
def get_captured_image():
    return send_file(os.path.join(captured_dir, "captured.jpg"))


@app.route("/get_detected_image")
def get_detected_image():
    return send_file(os.path.join(detected_dir, "detected.jpg"))


global_frame_cam1 = None
global_frame_cam2 = None
frame_lock = threading.Lock()


def oak_frame_reader(q, buffer_name):
    global global_frame_cam1, global_frame_cam2
    while True:
        in_frame = q.get()
        frame = in_frame.getCvFrame()
        with frame_lock:
            if buffer_name == "cam1":
                global_frame_cam1 = frame.copy()
            else:
                global_frame_cam2 = frame.copy()
        time.sleep(0.01)


if __name__ == "__main__":
    mxid_cam1 = "194430108191152F00"
    mxid_cam2 = "1944301021AADB2C00"

    # CAM 1
    pipeline1 = dai.Pipeline()
    cam1 = pipeline1.createColorCamera()
    cam1.setPreviewSize(640, 480)
    cam1.setInterleaved(False)
    xout1 = pipeline1.createXLinkOut()
    xout1.setStreamName("video")
    cam1.preview.link(xout1.input)

    device1 = dai.Device(pipeline1, dai.DeviceInfo(mxid_cam1))
    q1 = device1.getOutputQueue("video", maxSize=4, blocking=False)
    threading.Thread(target=oak_frame_reader, args=(q1, "cam1"), daemon=True).start()

    # CAM 2
    pipeline2 = dai.Pipeline()
    cam2 = pipeline2.createColorCamera()
    cam2.setPreviewSize(640, 480)
    cam2.setInterleaved(False)
    xout2 = pipeline2.createXLinkOut()
    xout2.setStreamName("video")
    cam2.preview.link(xout2.input)

    device2 = dai.Device(pipeline2, dai.DeviceInfo(mxid_cam2))
    q2 = device2.getOutputQueue("video", maxSize=4, blocking=False)
    threading.Thread(target=oak_frame_reader, args=(q2, "cam2"), daemon=True).start()

    app.run(debug=False, use_reloader=False)
