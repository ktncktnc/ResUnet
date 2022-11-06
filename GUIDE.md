# Dataset
- Alabama dataset: https://www.kaggle.com/datasets/meowmeowplus/alabama-buildings-segmentation
- S2Looking dataset: https://github.com/S2Looking/Dataset
- S2Looking color-correction:
    - Code: https://github.com/ktncktnc/ImageFuns/blob/master/image_hist.py
    - Steps:
        - Generate image histograms
        ```
        python image_hist.py --no-colorbalancing -s <path-to-images-folder> -o <path-to-histogram-folder>
        Ex: 
        - For alabama dataset
        python image_hist.py --no-colorbalancing -s /root/data/Alabama/image -o /root/data/Alabama/hist

        - For S2Looking dataset (train, test, val image1 and image2 folders)
        python image_hist.py --no-colorbalancing -s /root/data/S2Looking/train/Image1 -o /root/data/S2Looking/train/hist1
        ```
        - Color correction:
        python image_hist.py -s <source-image-folder> -sh <source-hist-folder> -r <ref-image-folder> -rh <ref-hist-folder> -o <output-folder>

        Ex:
        python image_hist.py -s /root/data/S2Looking/train/Image1 -sh /root/data/S2Looking/train/hist1 -r /root/data/Alabama/image -rh /root/data/Alabama/hist -o /root/data/S2Looking_cr/train/Image1
        ```