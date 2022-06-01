#!/usr/bin/env python3
import pyfirmata
import time
import pyfirmata2


if __name__ == "__main__":

    PORT = pyfirmata2.Arduino.AUTODETECT
    board = pyfirmata2.Arduino(PORT)
    servo = board.get_pin("d:9:s")  # Digital, Pin 9, Servo
    angle = 0
    while True:
        servo.write(angle)
        time.sleep(0.2)
        angle = (angle + 1) % 180
