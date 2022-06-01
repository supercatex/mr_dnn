import socket
from machine import Pin
import network
import esp
import gc
import time

esp.osdebug(None)
gc.collect()

ssid = "YOUR WIFI NAME"
password = "YOUR WIFI PASSWORD"
station = network.WLAN(network.STA_IF)
station.active(True)
station.connect(ssid, password)

led_B = Pin(17, Pin.OUT, value=0)
led_R = Pin(16, Pin.OUT, value=0)
while station.isconnected() == False:
    time.sleep(0.1)
    led_R.value(not led_R.value())
led_R.value(1)
print(station.ifconfig())

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 80))
s.listen(5)

while True:
    conn, addr = s.accept()
    print("Got a connection from %s" % str(addr))
    
    request = conn.recv(1024)
    request = str(request)
    print("Content = %s" % request)
    
    led_on = request.find('/?led=on')
    led_off = request.find('/?led=off')
    if led_on > 0:
        print("LED ON")
        led_B.value(1)
    if led_off > 0:
        print("LED OFF")
        led_B.value(0)
    
    response = "OK"
    conn.send("HTTP/1.1 200 OK\n")
    conn.send("Content-Type: text/html\n")
    conn.send("Connection: close\n\n")
    conn.sendall(response)
    conn.close()

led_R.value(0)
