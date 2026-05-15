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

### 3. Download required models

**請在第一次執行 `python main.py` 之前完成下列下載。** 僅下載 VGGT / SAM3 / SAM3D 主倉庫仍不夠：進入 **SAM3D 子進程**（3D 網格生成）時還會載入 **MoGe** 與 **DINOv2** 權重；若未預先下載且無法連線 Hugging Face，會出現類似 `Ruicheng/moge-vitl/resolve/main/model.pt` 的 `Network is unreachable` / `MaxRetryError`。

| 模型 | 來源 | 用途 | 建議本地路徑 / 備註 |
|------|------|------|---------------------|
| [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B) | HF / ModelScope | 場景幾何 | `models/VGGT` |
| [facebook/sam3](https://huggingface.co/facebook/sam3) | HF / ModelScope | 分割與追蹤 | `models/SAM3`（需含 `sam3.pt`） |
| [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects) | HF（需申請權限） | 3D 生成 checkpoint | `models/SAM3D` |
| **[Ruicheng/moge-vitl](https://huggingface.co/Ruicheng/moge-vitl)** | HF / ModelScope | SAM3D **depth_model**（見 `models/SAM3D/checkpoints/pipeline.yaml`） | 預設執行時從 HF 拉取 `model.pt` |
| **dinov2_vitl14_reg** | `torch.hub` | SAM3D backbone（`ss_generator.yaml` / `slat_generator.yaml`） | 需本地 clone + 權重快取（見下方） |

執行時若看到 `No rgb pointmap normalizer provided, using scale + shift`，屬 SAM3D 預處理提示，**不是**缺模型；需處理的是對 `huggingface.co` 的下載失敗。

#### 3-A. Hugging Face（需先 `hf auth login`，`sam-3d-objects` 需先在 HF 申請通過）

```bash
mkdir -p models
pip install 'huggingface-hub[cli]<1.0'

# 主流程三件套
hf download facebook/VGGT-1B --local-dir models/VGGT
hf download facebook/sam3 --local-dir models/SAM3
hf download facebook/sam-3d-objects --local-dir models/SAM3D

# SAM3D depth_model（未下載則子進程啟動時線上拉 model.pt）
hf download Ruicheng/moge-vitl
# 若要用固定目錄：hf download Ruicheng/moge-vitl --local-dir models/MoGe-vitl 並改 pipeline.yaml（見下方）
```

下載 MoGe 後可二選一避免線上請求：

- **方式 1（推薦，不改 yaml）**：只寫入 Hub 快取，勿用 `--local-dir`：
  ```bash
  hf download Ruicheng/moge-vitl
  ```
  保持 `pipeline.yaml` 中 `pretrained_model_name_or_path: Ruicheng/moge-vitl`。離線時可設 `export HF_HUB_OFFLINE=1`。
- **方式 2（固定到 `models/`）**：使用上文 `--local-dir models/MoGe-vitl`，並將 `pipeline.yaml` 第 65 行改為 `pretrained_model_name_or_path: ../../MoGe-vitl`（相對 `checkpoints/` → `models/MoGe-vitl`，目錄內需含 `model.pt`）。

#### 3-B. ModelScope（無法直連 Hugging Face 時）

```bash
mkdir -p models
modelscope download --model facebook/VGGT-1B --local_dir models/VGGT
modelscope download --model facebook/sam3 --local_dir models/SAM3
modelscope download --model facebook/sam-3d-objects --local_dir models/SAM3D

# MoGe（鏡像名稱可能與 HF 不同，下載後請確認目錄內有 model.pt，並按 3-A 方式 2 改 pipeline.yaml）
modelscope download --model bluestone25/moge-vitl --local_dir models/MoGe-vitl
```

#### 3-C. DINOv2 原始碼與權重（SAM3D 每次子進程都會載入）

```bash
# 1. 只 clone 一次（避免每次 torch.hub 同步 GitHub）
git clone https://github.com/facebookresearch/dinov2.git ~/dinov2

# 2. 每次跑 main.py 前（可寫進 ~/.bashrc 或 conda activate）
export SAM3D_DINOV2_LOCAL_REPO=$HOME/dinov2
# 或：export DINOV2_LOCAL_REPO=$HOME/dinov2

# 3. 固定 torch.hub 快取（權重預設在 checkpoints/ 下）
export TORCH_HUB_DIR=$HOME/.cache/torch/hub
mkdir -p "$TORCH_HUB_DIR"

# 4. （建議）預先拉取 dinov2_vitl14_reg 權重，避免 SAM3D 子進程首次啟動再下載
python - <<'PY'
import os, torch
from pathlib import Path
repo = Path(os.environ.get("SAM3D_DINOV2_LOCAL_REPO", os.path.expanduser("~/dinov2"))).resolve()
assert (repo / "hubconf.py").is_file(), f"Missing hubconf.py in {repo}"
hub = os.environ.get("TORCH_HUB_DIR")
if hub:
    torch.hub.set_dir(str(Path(hub).expanduser().resolve()))
torch.hub.load(str(repo), "dinov2_vitl14_reg", source="local", verbose=True)
print("DINOv2 vitl14_reg weights cached under:", torch.hub.get_dir())
PY
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
- Stage4 可選參數、輸出 JSON 與當前局限性說明見下方 **Run Examples (4080 Super 32 GB VRAM)** 小節。

## 💻Run Examples(4080 Super 32 GB VRAM)

```bash
python main.py --input_video ./assets/example/hallway.mp4 --output_path ./outputs/hallway --category_path ./assets/example/hallway.json --max_frames 80
```

啟用 Stage4 時可在上述命令末尾追加，例如：

```bash
python main.py ... --max_frames 80 --stage4 --stage4_yaw_range_deg 40 --stage4_n_yaw 17
```

**通用參數**

- `--input_video`: Path to the input video file or a directory containing image frames.
- `--output_path`: Directory where the output results will be saved.
- `--category_path`: Path to the JSON file containing category and relation information for the scene.
- `--max_frames`: Maximum number of frames to process from the video. The default value is set to 80 for a GPU with 32GB VRAM. You can adjust this value based on your hardware capabilities.
- `--no_center_mesh_at_geometry`: 預設會把每個重建網格平移到**幾何中心**為局部原點並同步更新 `T`；加上此旗標則維持 SAM3D 原始局部座標（原點多在場景原點附近）。

**Stage4 可選參數**（需加 `--stage4`；實作見 `src/Stage4Refiner.py`）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--stage4` | 關閉 | 在 SAM3D 粗對齊之後、Stage5 空間精化之前，對每個實例在**最佳視角幀**上做簡化視覺–空間對齊。 |
| `--stage4_yaw_range_deg` | `40.0` | 繞**世界 Z 軸**、過物體幾何中心的 yaw 搜尋半徑（度），實際搜尋區間為 `[-range, +range]`。 |
| `--stage4_n_yaw` | `17` | 在上述區間內的 yaw 候選個數（含端點）；建議為**奇數**以便包含 0°（當前粗對齊）。 |
| `--stage4_vertex_samples` | `2048` | 從網格隨機抽樣的頂點數，用於投影到影像並計算 mask 內點比例。 |
| `--stage4_seed` | `0` | 頂點隨機抽樣種子（與 `frame_id` 疊加），便於重現同一抽樣。 |

啟用 `--stage4` 後，會在 `{output_path}/instance_transforms_stage4.json` 寫入每個實例的 `origin_transform`（Stage4 前 `T`）與 `stage4_transform`（Stage4 後 `T`），便於對照。

**Stage4 當前實作與局限性**

本倉庫中的 Stage4 是**精簡占位實作**，不等同於論文中的完整 *render–match–optimize*（無可微渲染、無多視角光度損失、無平移/尺度聯合優化）。具體限制包括：

1. **僅優化繞世界 Z 的 yaw**：不調整平移、俯仰、滾轉或尺度；若誤差主要來自位置或其它旋轉，開 Stage4 幫助有限，少數情況會**看起來更歪**。
2. **打分準則較弱**：在最佳幀上將抽樣頂點投影到影像，以「落在實例 mask 內的比例」選最佳 yaw；該分數與人眼認定的對齊並非一一對應，對稱物、遮擋、細長/薄片網格可能選到錯誤朝向。
3. **僅當分數明顯變好才更新 `T`**：若所有候選 yaw 的 mask 內點比例都不優於當前姿態（閾值約 `1e-4`），則**不修改**——因此多數物件在 `instance_transforms_stage4.json` 中兩個矩陣幾乎相同屬正常現象。
4. **單幀、單 mask**：只使用 `all_optimal_frame_ids` 對應幀的 mask；VGGT 內參/影像尺寸與 mask 解析度不一致時會做 UV 縮放，仍可能存在對齊殘差。
5. **旋轉中心為網格包圍盒中心**：未必等於語義物體中心，大角度 yaw 時位移感可能被放大。

若觀察到少數物件 Stage4 後偏移變大，可先縮小 `--stage4_yaw_range_deg` 或關閉 `--stage4`；需要更穩定的對齊需擴展 Stage4（例如多幀約束、邊界距離損失、僅在分數增益超過閾值時更新等）。


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

