import sys

from pathlib import Path

def parse_csv(file_name):

    file_path = Path(file_name)

    with open (file_name, "r") as f:
        lines = f.readlines()

        with open(str(file_path.parent) + "/001.csv", "w") as f1:
            f1.write(lines[1])
            f1.write(lines[2])
            f1.write(lines[3])
            f1.write(lines[5])
            f1.write(lines[6])
            f1.write(lines[7])

        with open(str(file_path.parent) + "/002.csv", "w") as f2:
            f2.write(lines[10])
            f2.write(lines[11])
            f2.write(lines[12])
            f2.write(lines[14])
            f2.write(lines[15])
            f2.write(lines[16])

        with open(str(file_path.parent) + "/003.csv", "w") as f3:
            f3.write(lines[19])
            f3.write(lines[20])
            f3.write(lines[21])
            f3.write(lines[23])
            f3.write(lines[24])
            f3.write(lines[25])

        with open(str(file_path.parent) + "/004.csv", "w") as f4:
            f4.write(lines[28])
            f4.write(lines[29])
            f4.write(lines[30])
            f4.write(lines[32])
            f4.write(lines[33])
            f4.write(lines[34])


if __name__ == "__main__":
    parse_csv(sys.argv[1])