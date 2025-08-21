import Jetson.GPIO as GPIO
import time

led_pin = 7

GPIO.setmode(GPIO.BOARD)

GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW)

try:
    while True:
        GPIO.output(led_pin, GPIO.HIGH)
        print("LED is ON")
        time.sleep(3)

        GPIO.output(led_pin, GPIO.LOW)
        print("LED is OFF")
        time.sleep(3)

except KeyboardInterrupt:
    GPIO.cleanup()
    print("Exiting...")
    
    
    