import csv
import os
import glob

# Directory containing the CSV files
folder_path = '/mnt/HoudiniSamsungSs/Halo/Downloads/test2'

# Path where the combined CSV will be stored
combined_csv_name = 'attributes_iso.csv'
combined_csv_path = os.path.join('/mnt/HoudiniSamsungSs/Halo/mnist', combined_csv_name)

# Expected attribute names in the correct order
att_names = [
    "altitude", 
    "angle",
    "avg3x", "avg3y", "avg3z",
    "avghsvx", "avghsvy", "avghsvz",
    "avgx", "avgy", "avgz",
    "avgxyzx", "avgxyzy", "avgxyzz",
    "azimuth",
    "Cdx", "Cdy", "Cdz",
    "domnormx","domnormy","domnormz",
    "domx","domy","domz",
    "dot",
    "dot_eyedir_sundir",
    "eulerx", "eulery", "eulerz",
    "eyedirx", "eyediry", "eyedirz",
    "frame",
    #"height",
    "hsvx", "hsvy", "hsvz",
    "navgx", "navgy", "navgz",
    "orientw", "orientx", "orienty", "orientz",
    "qdot",
    "ralti",
    "rangle",
    "razi",
    "rcrossx","rcrossy","rcrossz",
    "rdot",
    "rdot_eyedir_sundir",
    "reulerx", "reulery", "reulerz",
    "reyedirx", "reyediry", "reyedirz",
    "rncrossx","rncrossy","rncrossz",
    "rorientw", "rorientx", "rorienty", "rorientz",  
    "rsundirx", "rsundiry", "rsundirz",  
    "sundirx", "sundiry", "sundirz",
    "xyzx", "xyzy", "xyzz"
]

# Initialize a list to hold all rows
all_rows = []

print("\nSTART\n")
# Load each CSV file in the folder path and print the values of every attribute
for filename in glob.glob(os.path.join(folder_path, '*.csv')):
    print(f"Reading {filename}")
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ordered_row = {att: row.get(att, '') for att in att_names}
            all_rows.append(ordered_row)
            # Print each row
            print(ordered_row)

# Write the combined CSV
with open(combined_csv_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=att_names)
    writer.writeheader()
    writer.writerows(all_rows)

print("\nDONE\n")
