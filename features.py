import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils import process

SAVE_PATH = Path(os.getcwd())

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clinical_file_path",
        required=True,
        type = str,
        help="route to raw clinical files"
    )
    
    parser.add_argument(
        "--accel_device_type",
        required=True,
        default="hip",
        help="raw accel device type: hip or wrist"
    )
    
    parser.add_argument(
        "--accel_files_path",
        required=True,
        help = "root path for raw accel files"
    )
    
    args, _ = parser.parse_known_args()
    dict_args = vars(args)
    
    clinical_df = pd.read_csv(dict_args["clinical_file_path"])
    
    output = {}
    source_path = list(Path(dict_args["accel_files_path"]).glob("*"))
    
    for raw_accel_path in tqdm(source_path):
        features = process(
            clinical_df = clinical_df,
            path = raw_accel_path,
            type = dict_args["accel_device_type"]
        )
        output[len(output)] = features
    
    output = pd.DataFrame.from_dict(output)
     
    output.to_csv(SAVE_PATH / "output.csv", index=False)
    
    return

if __name__ == "__main__":
    
    main()
