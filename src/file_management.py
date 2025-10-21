import os

def write_output_header(file_name):

    try:
        os.remove(file_name)
    except FileNotFoundError:
        print("output.txt does not yet exist")

    with open(file_name, "w") as f:
        f.write("Index,Status,Latitude,Longitude,Altitude,Time\n")

    return

def write_output(file_name, sample, status):

    with open(file_name, "a") as f:
        f.write(f"{sample['index']},{status},{sample['latitude']:.2f},{sample['longitude']:.2f},{sample['altitude']:.1f},{sample['time']}\n")

    return

def write_output_header_contrail(file_name):

    try:
        os.remove(file_name)
    except FileNotFoundError:
        print("output.txt does not yet exist")

    with open(file_name, "w") as f:
        f.write("Index,Status,Latitude,Longitude,Altitude,Time,Age,J_per_m,Avg_r(um),Avg_n(um),Avd_od\n")


    return

def write_output_contrail(file_name, sample, status, age, j_per_m, avg_r, avg_n, avg_od):

    with open(file_name, "a") as f:
        f.write(f"{sample['index']},{status},{sample['latitude']:.2f},{sample['longitude']:.2f},{sample['altitude']:.1f},{sample['time']},{age},{j_per_m},{avg_r},{avg_n},{avg_od}\n")

    return
