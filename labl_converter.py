import os

def modify_yolo_labels(txt_file_path):
    # Open the TXT file for reading and read all lines
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

    # Modify the lines to replace class 1 with class 0
    for i, line in enumerate(lines):
        class_id, *coords = line.split()
        if class_id == '1':
            lines[i] = '0 ' + ' '.join(coords) + '\n'

    # Open the TXT file for writing and write the modified lines back
    with open(txt_file_path, 'w') as f:
        f.writelines(lines)

def process_txt_files(folder_path):
    # Search for all TXT files in the folder
    for file in os.listdir(folder_path):

        if file.endswith('.txt'):
            # Construct the full path to the TXT file
            txt_file_path = os.path.join(folder_path, file)

            # Modify the TXT file using the modify_yolo_labels function
            modify_yolo_labels(txt_file_path)




process_txt_files('/home/mmt-ben/YOLOV8_VITIFER/custom dataset bud e_l/labels/val')
