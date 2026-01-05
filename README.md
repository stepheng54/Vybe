# Vybe
A music similarity model that finds songs that sound perceptually similar.

## Usage
This approach is most efficient when using the FMA dataset, but can be used with a custom dataset as well but the results are slightly less accurate and a little slower to produce.
If using FMA dataset, download all the associated CSV files, then run new_index.py and new_library.py. Once complete, can run search.py with an input song to search.
If using a custom dataset, make a folder for raw and processed data, then run prep_data.py, followed by search.py to search.
