# This code finds all images of patients of a specified VIRUS and X-Ray view and stores selected image to an OUTPUT directory
# + It uses metadata.csv for searching and retrieving images name
# + Using ./images folder it selects the retrieved images and copies them in output folder
# Code can be modified for any combination of selection of images
#

import pandas as pd
import shutil
import os

# Selecting all combination of 'COVID-19' patients with 'PA' X-Ray view
virus = "COVID-19" # Virus to look for
x_ray_view = "PA" # View of X-Ray

# metadata = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/metadata.csv" # Meta info
metadata = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/metadata_trimmed.csv" # Meta info
imageDir = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/images"   #  "../images" # Directory of images
# outputDir = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/covid19_imgs"     # '../output' # Output directory to store selected images
outputDir = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/non_covid19_imgs"     # '../output' # Output directory to store selected images

metadata_csv = pd.read_csv(metadata)

# loop over the rows of the COVID-19 data frame
for (i, row) in metadata_csv.iterrows():
	# if row["finding"] != virus or row["view"] != x_ray_view:             # Now extract only covid19 imgs
	if row["finding"] == virus or row["view"] != x_ray_view:             # Now extract only non-covid19 imgs
		continue

	filename = row["filename"].split(os.path.sep)[-1]
	filePath = os.path.sep.join([imageDir, filename])
	shutil.copy2(filePath, outputDir)
