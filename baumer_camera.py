# baumer_camera.py
import neoapi
import cv2

class BaumerCamera:
    def __init__(self, serial_number='700009456923'):
        self.camera = None
        self.serial_number = serial_number
        self.connect()

    def connect(self):
        try:
            self.camera = neoapi.Cam()
            self.camera.Connect(self.serial_number)
            self.camera.f.ExposureTime.Set(10000)  # default exposure
            print('Camera Connected!')
        except Exception as e:
            print(f"Error connecting to Baumer camera: {e}")
            self.camera = None

    def is_connected(self):
        return self.camera is not None and self.camera.IsConnected()

    def get_frame(self):
        if not self.is_connected():
            print("Camera not connected.")
            return None
        try:
            img = self.camera.GetImage()
            if not img.IsEmpty():
                return img.GetNPArray()
        except Exception as e:
            print(f"Baumer frame capture error: {e}")
        return None

    def get_exposure(self):
        if self.is_connected():
            try:
                return self.camera.f.ExposureTime.Get()
            except Exception as e:
                print(f"Failed to get exposure: {e}")
        return None

    def set_exposure(self, exposure_value):
        if self.is_connected():
            try:
                self.camera.f.ExposureTime.Set(exposure_value)
                return True
            except Exception as e:
                print(f"Failed to set exposure: {e}")
        return False
