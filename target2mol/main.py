import sys
import argparse
from data.cleaner import extract_drug_target_activity_data

if __name__ == '__main__':
    command = sys.argv[1]    
    
    if command == 'clean_data':
        extract_drug_target_activity_data()
