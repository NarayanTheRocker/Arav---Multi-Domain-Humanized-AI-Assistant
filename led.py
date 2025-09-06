import RPi.GPIO as GPIO
import time

# Pin Definitions
BLUE_LED_PIN = 27

# Set up the GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False) # Disable warnings for a cleaner output

# Set the pin as an output
GPIO.setup(BLUE_LED_PIN, GPIO.OUT)

print("Starting LED test...")
print("The blue LED should turn ON for 5 seconds and then turn OFF.")

try:
    # Turn the LED ON
    GPIO.output(BLUE_LED_PIN, GPIO.HIGH)
    print("Blue LED is now ON.")
    time.sleep(5)  # Wait for 5 seconds

    # Turn the LED OFF
    GPIO.output(BLUE_LED_PIN, GPIO.LOW)
    print("Blue LED is now OFF.")

except KeyboardInterrupt:
    print("Test interrupted by user.")
finally:
    # Clean up GPIO settings to reset the pins
    GPIO.cleanup()
    print("GPIO cleanup complete.")