import os
os.environ["LIDRA_SKIP_INIT"] = "true"
import argparse
import torch
import cv2
import numpy as np
import json
import trimesh
import sys

from src.models import load_vggt_model, load_sam3_image_model, load_sam3_video_model, unload_model
from src.utils import load_video_frames, vis_instance_masks
from src.geometry_utils import align_to_room_coordinate_system, align_vggt_predictions, get_optimal_view_frame_id, get_walls_info, apply_mesh_geometry_center_to_local_origin
from src.vggt_predict import vggt_predict
from src.object_segmentation import segment_wall_and_floor, segment_and_track
from src.sg_deduplication import self_category_deduplicate, cross_category_deduplicate
from src.instance_generation import generate_3d_asset_in_subprocess, generate_3d_asset
from src.sp_refinement import refine_supported_by_floor_object, refine_attached_to_wall_object, refine_embedded_in_wall_object
from src.pipeline_progress import cuda_cache_clear, cuda_mem_gib, log, timed_stage

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(
        f"[ReplicateAnyScene] 啟動 pipeline | device={device} | max_frames={args.max_frames} | "
        f"{cuda_mem_gib()}"
    )
    log(
        "[ReplicateAnyScene] 複雜度提示：幀數 N 上升時，"
        "「SAM3 牆/地」約為 2×N 次圖像推理；"
        "「SAM3 video」對每個類別各跑一次全長 propagate，耗時通常遠大於 80→160 的線性倍率。"
    )
    # Stage 1: Progressive object discovery
    # The code of this stage is not publicly available for now. 
    # You can refer to ./assets/example/hallway.json to manually specify object categories for the scene.
    with timed_stage("讀取類別 JSON", args.category_path):
        with open(args.category_path, 'r') as f:
            categories_and_relations = json.load(f)
    detected_categories = list(categories_and_relations.keys())
    print(f"Detected categories: {detected_categories}")

    # Stage 2: Spatial-Guided Visual Deduplication

    # Use vggt to predict 3d attributes
    with timed_stage("載入並預處理影片幀", f"max_frames={args.max_frames}"):
        frames = load_video_frames(args.input_video, args.max_frames).to(device)
    print(f"Loaded {len(frames)} frames for processing.")
    with timed_stage("VGGT 載入權重"):
        vggt_model = load_vggt_model().to(device)
    with timed_stage("VGGT 推論（單次前向，整段序列）"):
        vggt_prediction_results = vggt_predict(frames, vggt_model)
    del frames
    cuda_cache_clear("VGGT 後刪除 GPU 幀張量")

    vggt_model = unload_model(vggt_model)

    # Use sam3 to predict floor and wall to align the scene to room_coordinate_system
    with timed_stage(
        "SAM3 圖像：牆/地板分割",
        f"每幀 2 次 set_text_prompt，共 {len(vggt_prediction_results['colors'])} 幀",
    ):
        sam3_image_model = load_sam3_image_model()
        wall_masks, floor_masks = segment_wall_and_floor(vggt_prediction_results['colors'], sam3_image_model)
    with timed_stage("幾何對齊（房間座標系）"):
        R, t = align_to_room_coordinate_system(vggt_prediction_results['world_points'], wall_masks, floor_masks)
        vggt_prediction_results = align_vggt_predictions(vggt_prediction_results, R, t)
    # save related results
    with timed_stage("寫入中間結果（color/depth/extrinsics 等）"):
        os.makedirs(os.path.join(args.output_path, 'color'), exist_ok=True)
        os.makedirs(os.path.join(args.output_path, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(args.output_path, 'extrinsics'), exist_ok=True)
        np.savetxt(os.path.join(args.output_path, 'intrinsic.txt'), vggt_prediction_results['intrinsic'])
        vggt_prediction_results['point_cloud_data'].export(os.path.join(args.output_path, 'point_cloud.ply'))
        n_save = len(vggt_prediction_results['colors'])
        for i, image in enumerate(vggt_prediction_results['colors']):
            if i == 0 or (i + 1) % max(1, n_save // 8) == 0 or i + 1 == n_save:
                log(f"[ReplicateAnyScene] 寫入 color/depth/extrinsic 進度 {i+1}/{n_save}")
            cv2.imwrite(os.path.join(args.output_path, 'color', f"{i}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        for i, depth in enumerate(vggt_prediction_results['depths']):
            cv2.imwrite(os.path.join(args.output_path, 'depth', f"{i}.png"), (depth * 1000).astype(np.uint16)) # scale 1000
        for i, extrinsic in enumerate(vggt_prediction_results['extrinsics']):
            np.savetxt(os.path.join(args.output_path, 'extrinsics', f"{i}.txt"), extrinsic)
    cuda_cache_clear("寫盤完成")
    sam3_image_model = unload_model(sam3_image_model)
    # load sam3 video model and video frames
    with timed_stage("SAM3 video 載入與 start_session", f"幀數={len(vggt_prediction_results['colors'])}"):
        sam3_video_model = load_sam3_video_model()
        video_path = os.path.join(args.output_path, 'color')
        response = sam3_video_model.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
    session_id = response["session_id"]

    # for each category, segment and track the instances in the video
    with timed_stage("SAM3 video 逐類別追蹤", f"{len(detected_categories)} 個類別"):
        all_masks = {}
        for ci, category in enumerate(detected_categories):
            log(
                f"[ReplicateAnyScene] 類別進度 {ci+1}/{len(detected_categories)}: {category} | {cuda_mem_gib()}"
            )
            print(f"Segmenting and tracking category: {category}")
            category_masks = segment_and_track(category, sam3_video_model, session_id)
            # here we deduplicate inside each category
            deduplicated_category_masks = self_category_deduplicate(category_masks, vggt_prediction_results['world_points'], vggt_prediction_results['world_points_conf'])
            all_masks[category] = deduplicated_category_masks
    # here we deduplicate inside all objects
    with timed_stage("跨類別去重與可視化影片"):
        deduplicated_all_masks = cross_category_deduplicate(all_masks, vggt_prediction_results['world_points'], vggt_prediction_results['world_points_conf'])
        
        # vis the duplicated results
        vis_instance_masks(vggt_prediction_results['colors'], deduplicated_all_masks, os.path.join(args.output_path, 'instance_masks.mp4'))
    sam3_video_model = unload_model(sam3_video_model)
    cuda_cache_clear("SAM3 video 卸載後")

    # stage 3: Optimal-view asset generation

    # get optimal view frame id for each instance
    with timed_stage("計算每個實例最佳視角幀"):
        all_optimal_frame_ids = {}
        for category, category_masks in deduplicated_all_masks.items():
            all_optimal_frame_ids[category] = []
            for instance_masks in category_masks:
                optimal_frame_id = get_optimal_view_frame_id(vggt_prediction_results['world_points'], instance_masks)
                all_optimal_frame_ids[category].append(optimal_frame_id)

    # generate 3d assets for all instances in one SAM3D subprocess

    # If you meet ERROR: "RuntimeError: all_profile_res.empty() assert faild. can't find suitable algorithm for 0", use the code commented below.
    
    # sys.path.append('./sam-3d-objects/notebook')
    # from inference import Inference
    # inference = Inference(config_file="./models/SAM3D/checkpoints/pipeline.yaml", compile=False)
    # all_instances = {}
    # for category, category_masks in deduplicated_all_masks.items():
    #     all_instances[category] = []
    #     for instance_masks, optimal_frame_id in zip(category_masks, all_optimal_frame_ids[category]):
    #         image = vggt_prediction_results['colors'][optimal_frame_id]
    #         mask = next(im["mask"] for im in instance_masks if im["frame_id"] == optimal_frame_id)
    #         pointmap = vggt_prediction_results['world_points'][optimal_frame_id]
    #         extrinsic = vggt_prediction_results['extrinsics'][optimal_frame_id]
    #         print(f"Generating 3D asset for category: {category}, optimal frame id: {optimal_frame_id}")
    #         print(f"Image shape: {image.shape}, Mask shape: {mask.shape}, Pointmap shape: {pointmap.shape}, Extrinsic shape: {extrinsic.shape}")
    #         print(f"Image dtype: {image.dtype}, Mask dtype: {mask.dtype}, Pointmap dtype: {pointmap.dtype}, Extrinsic dtype: {extrinsic.dtype}")
    #         print(f"pixel num in mask: {(mask > 0).sum()}")
    #         try:
    #             instance_result = generate_3d_asset(image, mask, pointmap, extrinsic, inference)
    #         except Exception as e:
    #             print(f"Error occurred while generating 3D asset for category: {category}, optimal frame id: {optimal_frame_id}")
    #             print(f"Error: {e}")
    #             continue
    #         all_instances[category].append(instance_result)

    # We process in a subprocess to avoid strange CUDA error, but it may take a bit longer time.

    log(
        "[ReplicateAnyScene] SAM3D 子進程：單次 queue 等待最長 7200s；"
        "子進程內會列印 (done/total) 進度。"
    )
    with timed_stage("SAM3D 子進程：生成所有實例網格"):
        all_instances = generate_3d_asset_in_subprocess(
            deduplicated_all_masks,
            all_optimal_frame_ids,
            vggt_prediction_results['colors'],
            vggt_prediction_results['world_points'],
            vggt_prediction_results['extrinsics'],
        )
    cuda_cache_clear("SAM3D 子進程返回後（主進程）")

    # stage 4: Iterative Visual-Spatial Alignment
    # This part of the code is not publicly available for now.

    # stage 5: Semantic-Aware Scene Refinement
    # The code for this stage is not publicly available for now.
    # We provide the code for refining objects in "supported by floor", "attached to wall" and "embedded in wall" relationships.
    # You can refer to ./assets/example/hallway.json for the relationships of different categories in the scene.

    with timed_stage("空間精化（牆/地關係）與匯出 GLB"):
        walls_info = get_walls_info(vggt_prediction_results['world_points'], wall_masks)

        # We only process the "supported by floor", "attached to wall" and "embedded in wall" relationships in the current version
        for category, category_instances in all_instances.items():
            relationship = categories_and_relations[category]
            for instance_id, (optimal_frame_id, instance_info) in enumerate(zip(all_optimal_frame_ids[category], category_instances)):
                print(f"Refining {category}: {instance_id} with relationship: {relationship}")
                if relationship == "supported_by_floor":
                    instance_info = refine_supported_by_floor_object(instance_info)
                elif relationship == "embedded_in_wall":
                    instance_info = refine_embedded_in_wall_object(instance_info, walls_info)
                elif relationship == "attached_to_wall":
                    extrinsic = vggt_prediction_results['extrinsics'][optimal_frame_id]
                    camera_pos = - extrinsic[:3,:3].T @ extrinsic[:3,3]
                    instance_info = refine_attached_to_wall_object(instance_info, walls_info, camera_pos)
                else:
                    continue

        # save the final results（可選：網格局部原點置於幾何中心，再寫入 GLB）
        # 勿對網格先 apply_transform(T) 再 add_geometry：否則頂點已是世界座標、節點矩陣為單位，
        # DCC 裡「物件原點」仍會在場景原點，與幾何中心分離。應以 transform= 傳入 T，保留局部網格。
        scene = trimesh.Scene()
        for category, category_instances in all_instances.items():
            for i, instance_info in enumerate(category_instances):
                if not args.no_center_mesh_at_geometry:
                    apply_mesh_geometry_center_to_local_origin(instance_info)
                mesh = instance_info["original_mesh"]
                if mesh is None:
                    continue
                mesh_to_add = mesh.copy()
                T = np.asarray(instance_info["T"], dtype=np.float64)
                scene.add_geometry(
                    mesh_to_add,
                    node_name=f"{category}_{i}",
                    transform=T,
                )
        # z-up to y-up
        scene.apply_transform(np.array([[1, 0, 0, 0],[0, 0, 1, 0],[0, -1, 0, 0],[0, 0, 0, 1]]))
        scene.export(os.path.join(args.output_path, "final_scene.glb"))

    log(f"[ReplicateAnyScene] 全部完成 | 輸出: {args.output_path} | {cuda_mem_gib()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReplicateAnyScene pipeline")
    parser.add_argument("--input_video", type=str, default='./assets/example/hallway.mp4', help="Path to input video file or directory of images")
    parser.add_argument("--output_path", type=str, default='./outputs/hallway', help="Directory to save output results")
    parser.add_argument("--category_path", type=str, default='./assets/example/hallway.json', help="path to category and relation json")
    parser.add_argument("--max_frames", type=int, default=160, help="Maximum number of frames to process from the video")
    parser.add_argument(
        "--no_center_mesh_at_geometry",
        action="store_true",
        help="預設會把每個重建網格的頂點平移到幾何中心為局部原點並同步更新 T；"
        "若加此旗標則維持 SAM3D 原始局部座標（原點多在場景原點附近）。",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_video):
        raise FileNotFoundError(f"Input video or image directory not found: {args.input_video}")
    os.makedirs(args.output_path, exist_ok=True)
    main(args)