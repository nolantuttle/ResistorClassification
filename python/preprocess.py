#?------------------------------------------------------------
#? Author: Nolan Tuttle & Mathew Hobson
#? Project: ResistorClassification
#?------------------------------------------------------------

import os
import cv2 as cv
import numpy as np
import shutil as sh
import contour
from contour import isolate_band_region


def preprocess_dir(directory):
    new_directory = 'archive_clean'
    sh.copytree(directory, new_directory, dirs_exist_ok=True)

    num_files = sum(
        1 for root, dirs, files in os.walk(new_directory)
        for file in files if file.endswith('.jpg')
    )

    count = 0
    skipped = 0

    for root, dirs, files in os.walk(new_directory):
        for file in files:
            if not file.endswith('.jpg'):
                continue

            filepath = os.path.join(root, file)
            image = cv.imread(filepath)

            if image is None:
                skipped += 1
                os.remove(filepath)
                continue

            if count % 10 == 0:
                print(f'Files Processed: {count}')
                print(f'Percent Complete: {count / num_files * 100:.2f} %')

            try:
                band_crop, _ = isolate_band_region(filepath)
                band_crop = cv.resize(band_crop, (700, 700))
                cv.imwrite(filepath, band_crop)
            except Exception as e:
                skipped += 1
                os.remove(filepath)

            count += 1
            if count % 10 == 0:
                print(f'Files Processed: {count}')
                print(f'Percent Complete: {count / num_files * 100:.2f} %')

    print(f'\nDone. Processed: {count}, Skipped/Trashed: {skipped}')


preprocess_dir('archive')
