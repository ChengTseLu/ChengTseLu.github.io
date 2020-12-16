import board
import busio as io
import adafruit_mlx90614
import time

class thermal_sensor:

    def __init__(self):
        self.i2c = io.I2C(board.SCL, board.SDA, frequency=100000)
        self.mlx = adafruit_mlx90614.MLX90614(self.i2c)

    def temp(self):
        return self.mlx.ambient_temperature, self.mlx.object_temperature

if __name__ == "__main__":
    TS = thermal_sensor()
    for i in range(10):
        print(TS.temp())
        time.sleep(1)
