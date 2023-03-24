# tumor-budding-mil
Test code plus models for "Minimizing the intra- pathologist disagreement for tumor bud detection on H&amp;E images using weakly supervised learning" from SPIE Medical Imaging 2023

To run, one need to create .h5 files for each 512x512 ROI and extract 96x96 pixel patches from the ROI. Patches should be saved into an .h5 file with the named 'patches' along with a label for the ROI named 'label'.
