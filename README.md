# CoNR: Collaborative Neural Rendering using Anime Character Sheets

---

## [HomePage](https://conr.ml) | Colab | [arXiv](https://arxiv.org/abs/2207.05378)

![image](images/MAIN.png)


## Introduction

This project is the official implement of [Collaborative Neural Rendering using Anime Character Sheets](https://arxiv.org/abs/2207.05378), which aims to genarate vivid dancing videos from hand-drawn anime character sheets(ACS). Watch more demos in our [HomePage](https://conr.ml).
Contributors:

## Usage

#### Prerequisites

* NVIDIA GPU + CUDA + CUDNN
* Python 3.6

#### Installation

* Clone this repository

```bash
git clone https://github.com/megvii-research/CoNR
```

* Dependencies

To install all the dependencies, please run the following commands.

```bash
cd CoNR
pip install -r requirements.txt
```

* Download Weights
Download weights from Google Drive. Alternatively, you can download from [Baidu Netdisk](https://pan.baidu.com/s/1U11iIk-DiJodgCveSzB6ig?pwd=RDxc) (password:RDxc).

```
mkdir weights && cd weights
wget https://drive.google.com/file/d/1M1LEpx70tJ72AIV2TQKr6NE_7mJ7tLYx/view?usp=sharing
wget https://drive.google.com/file/d/1YvZy3NHkJ6gC3pq_j8agcbEJymHCwJy0/view?usp=sharing
wget https://drive.google.com/file/d/1AOWZxBvTo9nUf2_9Y7Xe27ZFQuPrnx9i/view?usp=sharing
wget https://drive.google.com/file/d/19jM1-GcqgGoE1bjmQycQw_vqD9C5e-Jm/view?usp=sharing
```

#### Prepare inputs
We prepared two Ultra-Dense Pose sequences for two characters, you can generate more UDPs via 3D models and motions. 
[Baidu Netdisk](https://pan.baidu.com/s/1hWvz4iQXnVTaTSb6vu1NBg?pwd=RDxc) (password:RDxc) 

```
# for short hair girl
wget https://drive.google.com/file/d/11HMSaEkN__QiAZSnCuaM6GI143xo62KO/view?usp=sharing
unzip short_hair.zip
mv short_hair/ poses/

# for double ponytail girl
wget https://drive.google.com/file/d/1WNnGVuU0ZLyEn04HzRKzITXqib1wwM4Q/view?usp=sharing
unzip double_ponytail.zip
mv double_ponytail/ poses/
```

We provide sample inputs of anime character sheets, you can also draw more by yourself.

```
# for short hair girl
wget https://drive.google.com/file/d/1r-3hUlENSWj81ve2IUPkRKNB81o9WrwT/view?usp=sharing
unzip short_hair_images.zip
mv short_hair_images/ character_sheet/

# for double ponytail girl
wget https://drive.google.com/file/d/1XMrJf9Lk_dWgXyTJhbEK2LZIXL9G3MWc/view?usp=sharing
unzip double_ponytail_images.zip
mv double_ponytail_images/ character_sheet/
```

#### RUN!
We provide two ways: with web UI or via terminal.

* with web UI (powered by [Streamlit](https://streamlit.io/))

```
streamlit run streamlit.py --server_port=8501
```
then open your browser and visit `localhost:8501`, follow the instructions to genarate video.

* via terminal

```
mkdir {dir_to_save_result}

python3 -m torch.distributed.launch \
--nproc_per_node=1 train.py --mode=test \
--world_size=1 --dataloaders=2 \
--test_input_poses_images={dir_to_poses} \
--test_input_person_images={dir_to_character_sheet} \
--test_output_dir={dir_to_save_result} \
--test_checkpoint_dir={dir_to_weights}

ffmpeg -r 30 -y -i {dir_to_save_result}/%d.png -r 30 -c:v libx264 output.mp4 -r 30
```

## Citation
```bibtex
@article{lin2022conr,
  title={Collaborative Neural Rendering using Anime Character Sheets},
  author={Lin, Zuzeng and Huang, Ailin and Huang, Zhewei and Hu, Chen and Zhou, Shuchang},
  journal={arXiv preprint arXiv:2207.05378},
  year={2022}
}
```

