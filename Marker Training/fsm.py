import rclpy
from rclpy.node import Node
from ur_msgs.msg import IOStates
from std_msgs.msg import String

import paho.mqtt.client as mqtt
import time

# MQTT broker details
broker = "c6288ac43b37439d9feeb57ac4cd3bc9.s1.eu.hivemq.cloud"   # e.g., "192.168.1.10" or "broker.hivemq.com"
port = 8883
display_topic = "feather/display"
image_id_topic = "feather/image_id"


# MQTT client setup
username = "ananya"
password = "Masdfab2425"
client = mqtt.Client()
client.username_pw_set(username, password)
client.tls_set()  # Use TLS for secure connection

client.connect(broker, port, 60)
# Wait for the connection to be established
client.loop_start()


class ToolButtonFSM(Node):
    """
    ROS node that runs a FSM (Finite State Machine) which monitors the buttons on the tool and manages the robot's modes.
    The FSM transitions between the following states:
    - IDLE: Waiting for the green button to be pressed.
    - FREEDRIVE_MODE: Activated when the green button is pressed, allowing free movement of the robot.
    - MODE_1_ALIGN: Activated when both green and red buttons are pressed, entering the design mode.
    - MODE_2_DESIGN: Activated when both buttons are pressed again, entering the alignment mode.    
    - MODE_3_MILL: Activated when both buttons are pressed again, entering the milling mode.     
    The FSM transitions back to IDLE when the green button is released.
    """
    def __init__(self):
        super().__init__('tool_button_fsm')
        # Define the pin for the green button
        global FREE_PIN
        FREE_PIN = 16  # This is the pin number for the green button
        global TCP_PIN
        TCP_PIN = 17  # This is the pin number for the red button
        self.subscription = self.create_subscription(
            IOStates,
            '/io_and_status_controller/io_states',
            self.io_callback,
            10)

        # Publisher for URScript command
        self.script_pub = self.create_publisher(String, '/urscript_interface/script_command', 10)

        self.state = "IDLE"
        self.green_pressed = False
        self.red_pressed = False

    def io_callback(self, msg: IOStates):
        """
        Callback function that processes the IOStates message to determine the state of the buttons.
        It updates the FSM based on the button states.
        """
        self.green_pressed = False
        self.red_pressed = False
        for din in msg.digital_in_states:
            if din.pin == FREE_PIN:
                self.green_pressed = din.state
                break
            elif din.pin == TCP_PIN:
                self.red_pressed = din.state
                break
        # Run the FSM step
        self.fsm_step()


    def fsm_step(self):
        r = self.red_pressed
        g = self.green_pressed
        s = self.state

        if s == "IDLE":
            if g and not r:
                self.state = "FREEDRIVE_MODE"
                self.activate_freedrive()

        elif s == "FREEDRIVE_MODE":
            if not g:
                self.state = "IDLE"
                self.deactivate_freedrive()
            elif g and r:
                self.state = "MODE_1_ALIGN"
                self.deactivate_freedrive()
                self.enter_mode_1()

        elif s == "MODE_1_ALIGN":
            if not g:
                self.state = "IDLE"
                self.exit_mode_1()
            elif g and r:
                self.state = "MODE_2_DESIGN"
                self.exit_mode_1()
                self.enter_mode_2()

        elif s == "MODE_2_DESIGN":
            if not g:
                self.state = "IDLE"
                self.exit_mode_2()
            elif g and r:
                self.state = "MODE_1_ALIGN"
                self.exit_mode_2()
                self.enter_mode_1()

    # --- Actions ---
    def activate_freedrive(self):
        self.get_logger().info("Activating freedrive mode with IO 17 condition")
        self.send_freedrive_script()
        client.publish(display_topic, "free!")

    def deactivate_freedrive(self):
        self.get_logger().info("Deactivating freedrive mode")
        client.publish(display_topic, "not free!")

    def enter_mode_1(self):
        self.get_logger().info("Entering Mode 1: Align")
        self.send_set_tcp_script()
        client.publish(display_topic, "align it!")

    def exit_mode_1(self):
        self.get_logger().info("Exiting Mode 1")
        client.publish(display_topic, "aligned!")

    def enter_mode_2(self):
        self.get_logger().info("Entering Mode 2: Design")
        self.send_set_tcp_script([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Set TCP to zero
        client.publish(display_topic, "design it!")

    def exit_mode_2(self):
        self.get_logger().info("Exiting Mode 2")
        client.publish(display_topic, "designed!")

    def send_freedrive_script(self):
        script = """
def keep_freedrive():
    freedrive_mode()
    while (get_tool_digital_in(0)):   
        sleep(0.1) 
    end
end
"""
        msg = String()
        msg.data = script
        self.script_pub.publish(msg)
        self.get_logger().info("Sent URScript to hold freedrive while IO 17 is on")
    
    def send_set_tcp_script(self, p=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        script = """
                def set_tcp():
                    set_tcp(p[%f, %f, %f, %f, %f, %f])])
                end
                """ % (p[0], p[1], p[2], p[3], p[4], p[5])
        msg = String()
        msg.data = script
        self.script_pub.publish(msg)
        self.get_logger().info("Sent URScript to set TCP")

def main(args=None):
    rclpy.init(args=args)
    node = ToolButtonFSM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

