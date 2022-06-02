#!/usr/bin/env python3
from paho.mqtt.client import Client


if __name__ == "__main__":

    client = Client()
    client.connect("YOUR IP ADDRESS", 1883, 60)

    while True:
        s = input()
        if s == "q": break
        client.publish("cmd", s)
