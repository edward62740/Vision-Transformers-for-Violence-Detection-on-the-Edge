"""
This script converts the spatiotemporal annotations from the UCFCrime2Local dataset to temporal annotations following
the format in the standard UCF-Crime dataset. The output is a text file with the following format:
<video_name> <crime_type> <start_frame1> <end_frame1> <start_frame2> <end_frame2>

"""

import os
from itertools import groupby
from operator import itemgetter

PROJ_DIR = r""

import os


def process_file(input_file, output_lines):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    filename = os.path.basename(input_file).split('.')[0]
    crime_type = lines[9].strip().split()[9].strip('"')
    anomaly_indices = [i for i, line in enumerate(lines) if line.strip().split()[6] == '0']
    anomaly_indices.sort()

    ranges = []
    for k, g in groupby(enumerate(anomaly_indices), lambda x: x[0] - x[1]):
        group = (map(itemgetter(1), g))
        group = list(map(int, group))
        ranges.append((group[0], group[-1]))

    if ranges.__len__() == 2:
        start_frame1 = ranges[0][0]
        end_frame1 = ranges[0][1]
        start_frame2 = ranges[1][0]
        end_frame2 = ranges[1][1]
        print(start_frame1, end_frame1, start_frame2, end_frame2)
        output_lines.append(
            f'{filename}_x264.mp4  {crime_type}  {start_frame1}  {end_frame1}  {start_frame2}  {end_frame2}')
    elif ranges.__len__() == 1:
        start_frame1 = ranges[0][0]
        end_frame1 = ranges[0][1]
        start_frame2 = -1
        end_frame2 = -1
        output_lines.append(
            f'{filename}_x264.mp4  {crime_type}  {start_frame1}  {end_frame1}  {start_frame2}  {end_frame2}')


def main(source_folder):
    output_lines = []
    for file in os.listdir(source_folder):
        if file.endswith('.txt'):
            input_file = os.path.join(source_folder, file)
            process_file(input_file, output_lines)

    with open("out.txt", 'w') as f:
        f.write('\n'.join(output_lines))


if __name__ == "__main__":
    source_folder = PROJ_DIR + r"/UCFCrime2Local/annotations"
    main(source_folder)
