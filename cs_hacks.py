import cv2
import numpy as np
from plyer import vibrator
from gtts import gTTS
import os
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout

class BlindNavApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cap = cv2.VideoCapture(0)  # Use the first camera
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.object_detected = False
        self.danger_detected = False
        self.distance_threshold = 50  # Threshold for object proximity

    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.feedback_label = Label(text="Waiting for input...", size_hint=(1, 0.1))
        self.nav_button = Button(text="Start Navigation", size_hint=(1, 0.1))
        self.nav_button.bind(on_press=self.start_navigation)

        self.layout.add_widget(self.feedback_label)
        self.layout.add_widget(self.nav_button)
        return self.layout

    def start_navigation(self, instance):
        self.feedback_label.text = "Starting Navigation..."
        self.run_navigation()

    def run_navigation(self):
        while True:
            # Capture frame from camera
            ret, frame = self.cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_color = np.array([30, 150, 50])  # Object color range (adjustable)
            upper_color = np.array([85, 255, 255])

            mask = cv2.inRange(hsv, lower_color, upper_color)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Detect objects
            for contour in contours:
                if cv2.contourArea(contour) > 1000:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Estimate object distance
                    object_distance = w  # Width used as a proxy for distance
                    if object_distance < self.distance_threshold:
                        self.object_detected = True
                        self.provide_vibration_feedback()
                        self.feedback_label.text = f"Object nearby at {object_distance}px!"
                        self.provide_audio_feedback(f"Object nearby at {object_distance} pixels!")

            # Check for fall hazards (edges of the frame)
            if self.detect_fall_hazard(frame):
                self.danger_detected = True
                #self.provide_vibration_feedback()
                self.feedback_label.text = "Warning: Fall hazard detected!"
                self.provide_audio_feedback("Warning: Fall hazard detected!")

            # Display the camera feed
            cv2.imshow("Camera Feed", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release camera and close windows
        self.cap.release()
        cv2.destroyAllWindows()

    def detect_fall_hazard(self, frame):
        # Check the bottom-left and bottom-right corners for fall hazards
        height, width, _ = frame.shape
        bottom_left = frame[height-50:, :50]  # Bottom-left corner
        bottom_right = frame[height-50:, width-50:]  # Bottom-right corner
        if np.mean(bottom_left) < 50 or np.mean(bottom_right) < 50:
            return True
        return False

   # def provide_vibration_feedback(self):
        vibrator.vibrate(1)  # Vibrate for 1 second

    def provide_audio_feedback(self, message):
        tts = gTTS(message, lang='en')
        tts.save("feedback.mp3")
        os.system("afplay feedback.mp3")  # Play the audio feedback


if __name__ == '__main__':
    BlindNavApp().run()
