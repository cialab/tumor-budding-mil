# tumor-budding-mil
Test code plus models for "Minimizing the intra- pathologist disagreement for tumor bud detection on H&amp;E images using weakly supervised learning" from SPIE Medical Imaging 2023

To run, create .h5 files for each 512x512 ROI and extract 96x96 pixel patches. Shape of resulting bag should be nx3x96x96, where n is the number of patches. Patches should be saved with the named 'patches' along with a label for the ROI named 'label'. 

Test script needs a directory where .h5 files are store. Each subdirectory should correspond to a slide, while each .h5 ROI is saved within their respective slide subdirectories. Test script will compute tumor budding probabilities and predictions for each ROI across an entire WSI and save them into a .h5 file. Will also compute AUC, precision, and recall based on provided ROI labels.
