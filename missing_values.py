"""
This script scans for missing values in:
	acceleration.csv
	*_AP.csv
and provides stats
"""

# import modules
import pandas as pd
import glob

# define files
acceleration_files = glob.glob('data/*/*/acceleration.csv')
ap_files = glob.glob('data/*/*/*_AP.csv')
print('there are ',len(acceleration_files),'acceleration files')
print('there are ',len(ap_files),'AP files')
total_count = 0
corrupted_files = []

# loop through files
for acceleration_file in acceleration_files:
	acceleration = pd.read_csv(acceleration_file)
	nrows,ncol = acceleration.shape
	count = nrows - acceleration.apply(lambda x: x.count(), axis=0)
	if count.isnull().values.any():
		corrupted_files.append(acceleration_file)
		continue
	total_count = total_count + count

print(total_count)
print('there are ',len(corrupted_files),' corrupted files')
print('these files are: ',corrupted_files)

	



