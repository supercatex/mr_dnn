import gc
import esp
import time
import network
from machine import Pin
from umqtt.simple import MQTTClient

esp.osdebug(None)
gc.collect()


# Variables
PIN16 = Pin(16, Pin.OUT, value=0)
PIN17 = Pin(17, Pin.OUT, value=0)
SSID, PSWD = "FablabHome2", "88887777"
HOST = "192.168.50.52"


# Connect WIFI
station = network.WLAN(network.STA_IF)
station.active(True)
station.connect(SSID, PSWD)

while not station.isconnected():
    time.sleep(0.1)
    PIN16.value(not PIN16.value())
PIN16.value(1)
print(station.ifconfig())


# MQTT
def mqtt_callback(topic, msg):
    global PIN17
    if msg == b"1": PIN17.value(1)
    if msg == b"0": PIN17.value(0)

client = MQTTClient("ESP32", HOST)
client.set_callback(mqtt_callback)
client.connect()
client.subscribe("cmd")


# MAIN LOOP
while True:
    time.sleep(0.05)
    client.check_msg()
