# File for programming Sabertooth commands on the wheelchair.

# !/usr/bin/env python
from pysabertooth import Sabertooth
import serial.tools.list_ports as port
import time
import tkinter as tk


def forward(speed):
    print("forward")
    # saber.text('m1:startup')
    # saber.text('m2:startup')
    saber.drive(1, speed)
    saber.drive(2, speed)


def backward(speed):
    print("backward")
    # saber.text('m1:startup')
    # saber.text('m2:startup')
    saber.drive(1, -speed)
    saber.drive(2, -speed)


def left(speed_angular):
    print("left")
    # saber.text('m1:startup')
    # saber.text('m2:startup')
    saber.drive(1, -speed_angular)
    saber.drive(2, speed_angular)


def right(speed_angular):
    print("right")
    # saber.text('m1:startup')
    # saber.text('m2:startup')
    saber.drive(1, speed_angular)
    saber.drive(2, -speed_angular)
    time.sleep(2)


def stop():
    print("stop")
    saber.drive(1, 0)
    saber.drive(2, 0)
    time.sleep(0.1)
    saber.stop()


def initialization():
    print("\nDetecting sabertooth....\n")
    pl = list(port.comports())
    print(pl)
    address = ""
    for p in pl:
        print(p)
        if "Sabertooth" in str(p):
            address = str(p).split(" ")
    print("\nAddress found @")
    print(address[0])

    saber = Sabertooth(address[0], baudrate=9600, address=128, timeout=0.1)

    return saber


def key(event):
    print('Special Key %r' % event.keysym)
    if event.keysym == 'Escape':
        stop()
        root.destroy()
    elif event.keysym == "Up":
        forward(speed)
    elif event.keysym == "Down":
        backward(speed)
    elif event.keysym == "Right":
        right(speed_angular)
    elif event.keysym == "Left":
        left(speed_angular)
    elif event.keysym == "s":
        stop()


if __name__ == '__main__':
    speed = 30
    speed_angular = 6
    saber = initialization()

    root = tk.Tk()
    print("Start!")
    root.bind_all('<Key>', key)
    # don't show the tk window
    # root.withdraw()
    root.mainloop()






