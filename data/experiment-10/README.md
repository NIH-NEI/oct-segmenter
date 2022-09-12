# Experiment 10

# Dataset Generation 
In order to generate the datasets, we performed the following steps:
1. Copied and compressed the data shared by the NEI VFC group in the "Full Frame" directory in Box.
The compressed file was named `experiment-10-images.tar.gz`.
2. Created a TSV file that maps the image file names to their corresponding subjects using the
script in `preprocessing-scripts/custom/map_image_name_to_subject.py`. The generated file was saved
in `data/experiment-10/filename_to_subject.tsv`.
3. Splitted the images into `train`, `test` and `validation` using the script
`preprocessing-scripts/custom/split_images_into_train_val_test.py`.
4. Used the `oct-segmenter` to create the `training_dataset.hdf5`:
```
python run.py generate training \
    -i data/experiment-10/images/training/ \
    -v data/experiment-10/images/validation/ \
    -m \
    -o data/experiment-10/
```

5. Used the `oct-segmenter` to create the `test_dataset.hdf5`:
```
python run.py generate test \
    -i data/experiment-10/images/test/ \
    -m \
    -o data/experiment-10/
```
