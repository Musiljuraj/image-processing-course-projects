from pynput.mouse import Controller
import time

"""
Manual helper tool. It continuously reads and prints current mouse coordinates using pynput.
With the WSL - Windows configuration its not relevant. 
ROI coordinates were in fact solved (received) at the Windows side of the project.
"""

mouse = Controller()

print("Move mouse to desired point. Press Ctrl+C to stop.")

try:
    while True:
        x, y = mouse.position
        print(f"\rX={x:4d}  Y={y:4d}", end="", flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nStopped.")