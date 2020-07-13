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

# # metadata = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/metadata.csv" # Meta info
# metadata = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/metadata_trimmed.csv" # Meta info
# metadata = "/home/pdp1145/chest_xray_imgs/Stanford_xrays/train.csv"  # Meta info
metadata = "/home/pdp1145/chest_xray_imgs/CheXpert-v1.0-small/train.csv"
         
# # imageDir = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/images"   #  "../images" # Directory of images
# imageDir = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/images"   #  "../images" # Directory of images
# imageDir = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/images"   #  "../images" # Directory of images
imageDir = "/home/pdp1145/chest_xray_imgs/CheXpert-v1.0-small/train"
image_pre_dir = "CheXpert-v1.0-small/train"

# outputDir = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/covid19_imgs"     # '../output' # Output directory to store selected images
outputDir = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/non_covid19_imgsx/"     # '../output' # Output directory to store selected images

metadata_csv = pd.read_csv(metadata)

# loop over the rows of the COVID-19 data frame
for (i, row) in metadata_csv.iterrows():
	# if row["finding"] != virus or row["view"] != x_ray_view:             # Now extract only covid19 imgs
	if row["AP/PA"] != x_ray_view or row["No Finding"] != 1:             # Now extract only non-covid19 imgs
		continue

	filename = row["Path"]   #  .split(os.path.sep)[-1]
	fpath, fname = filename.split(image_pre_dir)
	fname = fname[1:]
	ofname = fname.replace("/", "_")
	filePath = os.path.sep.join([imageDir, fname])
	ofpath = outputDir + ofname
	shutil.copy2(filePath, ofpath)
