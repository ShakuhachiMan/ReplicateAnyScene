# ✨ReplicateAnyScene: Zero-Shot Video-to-3D Composition via Textual-Visual-Spatial Alignment✨

[Mingyu Dong](https://dongmingyu111.github.io/)1,*, [Chong Xia](https://xiac20.github.io/)1,*, Mingyuan Jia1, [Weichen Lyu](https://matthew-lyu.github.io/)1, [Long Xu](https://gaolon.github.io/xulong/)2, [Zheng Zhu](https://www.zhengzhu.net/)1, [Yueqi Duan](https://duanyueqi.github.io/)1,†  
1Tsinghua University   2Zhejiang University

       

       

     Teaser Visualization

**ReplicateAnyScene:** We propose ReplicateAnyScene, a framework capable of fully automated and zero-shot transformation of casually captured videos into compositional 3D scenes.

## 📢 News

- 🔥 [04/14/2026] We release the code for stage 2 and 3, as well as partial code for stage 1 and 5.
- 🔥 [04/14/2026] We release "ReplicateAnyScene: Zero-Shot Video-to-3D Composition via Textual-Visual-Spatial Alignment". Check our [project page](https://xiac20.github.io/ReplicateAnyScene) and [arXiv paper](https://arxiv.org/abs/2604.10789).

## 🌟 Pipeline

Pipeline Visualization

**The overall framework of our approach ReplicateAnyScene.** Our pipeline consists of a five-stage cascade where each stage is specifically designed to resolve targeted alignment gaps among our three core modalities including **textual (green), visual (orange), and spatial (blue)**. The gradient backgrounds and multi-colored dashed borders within each module explicitly illustrate the specific cross-modal alignment process occurring at that step.

## ⚙️ Setup

### 1. Clone Repository and update submodules

```bash
git clone https://github.com/xiac20/ReplicateAnyScene.git
cd ReplicateAnyScene
git submodule update --init --recursive
```

### 2. Environment Setup

1. **Create conda environment**

```bash
conda env create -f environments/default.yml
conda deactivate
conda activate ReplicateAnyScene
```

1. **Install SAM3D-related dependencies**

```bash
cd sam-3d-objects
# for pytorch/cuda dependencies
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# install sam3d-objects and core dependencies
pip install -e '.[dev]'
pip install -e '.[p3d]' # pytorch3d dependency on pytorch is broken, this 2-step approach solves it

# for inference
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

# patch things that aren't yet in official pip packages
./patching/hydra # https://github.com/facebookresearch/hydra/pull/2863

cd ../ # back to root
```

1. **Install SAM3,VGGT,and other dependencies**

```bash
cd sam3
pip install .[dev,notebooks,train]
cd ../vggt
pip install -e .
cd ..
pip install colorcet
```

1-A. **Download required models(Huggingface).**

```bash
mkdir models
hf download facebook/VGGT-1B  --local-dir models/VGGT
hf download facebook/sam3 --local-dir models/SAM3
hf download facebook/sam-3d-objects --local-dir models/SAM3D
```

1-B. **Download required models(ModelScope).**

```bash
mkdir models
modelscope download --model facebook/VGGT-1B --local_dir models/VGGT
modelscope download --model facebook/sam3 --local_dir models/SAM3
modelscope download --model facebook/sam-3d-objects --local_dir models/SAM3D
```

1. **Download DINO V2 git source.**

```bash
# 1. 只 clone 一次（本機常駐）
git clone https://github.com/facebookresearch/dinov2.git ~/dinov2
# 2. 每次跑腳本前 export（可寫進 ~/.bashrc 或 conda activate 腳本）
export SAM3D_DINOV2_LOCAL_REPO=$HOME/dinov2
# 或通用名稱亦可：
# export DINOV2_LOCAL_REPO=$HOME/dinov2

# 3. （可選）固定 hub 快取目錄，方便備份/避免容器內重下
export TORCH_HUB_DIR=$HOME/.cache/torch/hub
```

## 💻Run Examples(Default)

！以下是原始执行命令，但是没有48GB VRAM 不推荐执行下面，我用32GB VRAM 等待了很久没有返回：  
We provide an example scene to help you get started.

```bash
python main.py --input_video ./assets/example/hallway.mp4 --output_path ./outputs/hallway --category_path ./assets/example/hallway.json --max_frames 160
```

- `--input_video`: Path to the input video file or a directory containing image frames.
- `--output_path`: Directory where the output results will be saved.
- `--category_path`: Path to the JSON file containing category and relation information for the scene.
- `--max_frames`: Maximum number of frames to process from the video. The default value is set to 160 for a GPU with 48GB VRAM. You can adjust this value based on your hardware capabilities.

## 💻Run Examples(4080 Super 32 GB VRAM)

```bash
python main.py --input_video ./assets/example/hallway.mp4 --output_path ./outputs/hallway --category_path ./assets/example/hallway.json --max_frames 80
```

- `--input_video`: Path to the input video file or a directory containing image frames.
- `--output_path`: Directory where the output results will be saved.
- `--category_path`: Path to the JSON file containing category and relation information for the scene.
- `--max_frames`: Maximum number of frames to process from the video. The default value is set to 80 for a GPU with 32GB VRAM. You can adjust this value based on your hardware capabilities.
- `--no_center_mesh_at_geometry`: 设置场景中心作为重建模型的坐标中心。

## 🔗Acknowledgement

We are thankful for the following great works when implementing SimRecon:

- [SimRecon](https://github.com/xiac20/SimRecon), [Spatial-MLLM](https://github.com/THU-SI/Spatial-MLLM), [SAM3](https://github.com/facebookresearch/sam3), [SAM3D](https://github.com/facebookresearch/sam-3d-objects), [VGGT](https://github.com/facebookresearch/vggt), [Qwen3VL](https://github.com/QwenLM/Qwen3-VL), [MASt3R](https://github.com/naver/mast3r)

## 📚Citation

```bibtex
@misc{dong2026replicateanyscenezeroshotvideoto3dcomposition,
      title={ReplicateAnyScene: Zero-Shot Video-to-3D Composition via Textual-Visual-Spatial Alignment}, 
      author={Mingyu Dong and Chong Xia and Mingyuan Jia and Weichen Lyu and Long Xu and Zheng Zhu and Yueqi Duan},
      year={2026},
      eprint={2604.10789},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.10789}, 
}
```

