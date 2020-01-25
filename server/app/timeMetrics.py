import time
import csv
from pathlib import Path

class timeMetrics:

    def __init__(self):
        super().__init__()
        self.times = {}

    def addTime(self, label):
        actualTime = time.time()
        self.times[label] = actualTime

    def addTimeMode1(self, label):
        actualTime = time.time()
        if hasattr(self.times, label) == False:
            self.times[label] = actualTime

    def printTimes(self):
        print(self.times)

    def resetTimes(self):
        self.times = {}

    def to_csv(self):
        try:
            PATHTOCSV = "Mode3TimeMetricsByClient.csv"
            my_file = Path(PATHTOCSV)
            if my_file.is_file():
                with open(PATHTOCSV, 'a', newline='') as csvfile:
                    writer2 = csv.DictWriter(csvfile, self.times.keys())
                    writer2.writerow(self.times)
            else:
                with open(PATHTOCSV, 'w', newline='') as csvfile:
                    writer2 = csv.DictWriter(csvfile, self.times.keys())
                    writer2.writeheader()
                    writer2.writerow(self.times)
        except IOError:
            print("I/O error")

    def to_csvMode1(self):
        try:
            PATHTOCSV = "Mode1TimeMetrics.csv"
            my_file = Path(PATHTOCSV)
            if my_file.is_file():
                with open(PATHTOCSV, 'a', newline='') as csvfile:
                    writer2 = csv.DictWriter(csvfile, self.times.keys())
                    writer2.writerow(self.times)
            else:
                with open(PATHTOCSV, 'w', newline='') as csvfile:
                    writer2 = csv.DictWriter(csvfile, self.times.keys())
                    writer2.writeheader()
                    writer2.writerow(self.times)
        except IOError:
            print("I/O error")
