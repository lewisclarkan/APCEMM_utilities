import os

TEMPLATE_FILE = "slurm_submit_final.peta4-cclake"
SUBMIT_FILE = "slurm_submit_final_submit.peta4-cclake"

N = 19999
step = 100
month = 1

for start in range(0,N,step):
    end = min(start+step-1,N-1)
    print(f"Processing indices {start}-{end}")

    with open(TEMPLATE_FILE, "r") as f:
        content = f.read()
    content = content.replace("{START}", str(start))
    content = content.replace("{END}", str(end))
    content = content.replace("{MONTH}", str(month))
    
    with open(SUBMIT_FILE, "w") as f:
        f.write(content)

    os.system(f"sbatch {SUBMIT_FILE}")
    os.remove(SUBMIT_FILE)
