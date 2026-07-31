#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import speech_recognition as sr

class GoogleSpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__('google_speech_recognition')
        self.publisher_=self.create_publisher(String, 'sr_output', 10)
        self.get_logger().info("Google Speech Recognition Node Starts Now")

    def listen_and_publish(self):
        recognizer= sr.Recognizer()

        with sr.Microphone() as source:
            self.get_logger().info("Say Something")

            try:

                audio= recognizer.record(source, duration=5)

                result= recognizer.recognize_google(audio, language="en-US")
                self.get_logger().info(f"SR Result: {result}")

                msg=String()
                msg.data= result
                self.publisher_.publish(msg)
            except sr.UnknownValueError:
                self.get_logger().info("SR could not understand the audio")
            except sr.RequestError as e:
                self.get_logger().info(f"Could Not request results from Google: {e}")
    
def main(args=None):
    rclpy.init(args=args)
    node= GoogleSpeechRecognitionNode()
    try:
        while rclpy.ok():
            node.listen_and_publish()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()
    
