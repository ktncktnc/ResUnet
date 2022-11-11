# Dataset
- Alabama dataset: 
    - Link: https://www.kaggle.com/datasets/meowmeowplus/alabama-buildings-segmentation
    - Train/test split: [Link](https://drive.google.com/drive/folders/1zM2_TnQ4DNe98DMx5b6sUmrrKEiJKvNi) Download and put those .txt files into alabama folder
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
        ```
        python image_hist.py -s <source-image-folder> -sh <source-hist-folder> -r <ref-image-folder> -rh <ref-hist-folder> -o <output-folder>

        Ex:
        python image_hist.py -s /root/data/S2Looking/train/Image1 -sh /root/data/S2Looking/train/hist1 -r /root/data/Alabama/image -rh /root/data/Alabama/hist -o /root/data/S2Looking_cr/train/Image1
        ```
# Segmentation
- Training:
    - Baseline:
    ```
    python train_2branchnet_segmentation_baseline.py -c <config-file> --epochs <epoch-number> --resume <optional: pretrained-path> --name <experiment-name> --device <device-name: cpu, cuda:0>
    ```
    - DASM and I-DASM: khác nhau ở chỗ config dataset trong file config
    ```
    python train_2branchnet_segmentation.py -c <config-file> --epochs <epoch-number> --resume <optional: pretrained-path> --name <experiment-name> --device <device-name: cpu, cuda>
- Inference:
    ```
    python instance_segmentation.py -c config <config-file> --pretrain <trained-file> --device <device-name: cpu, cuda> --split <split-name: train, test, val> --dset_divide <optional: divide an input image axis to n part> --savepath <result-folder> 
    ```
# Change detection
- Training
    - Baseline:
    ```
    python train_2branchnet_change_detection.py -c <config-path> --epochs <epochs> --resume <optional: pretrained-path> --name <experiment-name> --device <device-name: cpu, cuda:0> --no_prob_input
    ```
    - Add segmentation mask as the fourth channel:  Generate ảnh segmentation ở dạng prob bằng script instance_segmentation.py và copy vào thư mục prob_img1, prob_img2 ở mỗi split train, val, test
    ```
    python train_2branchnet_change_detection.py -c <config-path> --epochs <epochs> --resume <optional: pretrained-path> --name <experiment-name> --device <device-name: cpu, cuda:0>
    ```
- Testing:
    ```
    python change_detection_s2looking.py -c <config-path> --pretrain <trained-file> --device <device-name: cpu, cuda> --split <split-name: train, test, val> --dset_divide <optional: divide an input image axis to n part> --savepath <result-folder>(--segmentation_mask or --no_segmentation_mask)
    ```