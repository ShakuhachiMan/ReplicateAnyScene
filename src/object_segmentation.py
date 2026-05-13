import contextlib

import torch
from html import parser
import numpy as np
from PIL import Image

def segment_wall_and_floor(images, sam3_image_model):
    """
    Use SAM3 to segment wall and floor from the input images.
    
    Args:
        images: numpy array of shape (S, H, W, 3)
        sam3_image_model: The loaded SAM3 image model object.
    Returns:
        wall_masks: A list of dictionaries containing 'frame_id' and 'mask' (binary mask of the wall).
        floor_masks: A list of dictionaries containing 'frame_id' and 'mask' (binary mask of the floor).
        [
        
            {
                'frame_id': int,
                'mask': numpy array of shape (H, W) with binary values (True or False)
            },
            ...
        ]
    """
    # 因为使用的显卡不是 Amper 架构，需要精度转换 
    # 1. 图像预处理 
    #processed_image = processor.preprocess(image).to("cuda") 
 
    # 2. 自动判断GPU架构，选择最优对齐精度 
    #compute_capability = torch.cuda.get_device_capability()[0] 
    #if compute_capability >= 8:  # Ampere及以上架构 
    #    target_dtype = torch.bfloat16 
    #elif compute_capability >= 7:  # Turing架构 
    #    target_dtype = torch.float16 
    #else:  # 更旧架构 
    #    target_dtype = torch.float32 
 
    # 3. 模型和输入统一转换到目标精度（仅需执行一次，重复调用无开销） 
    #if sam3_image_model.dtype != target_dtype: 
    #    sam3_image_model = sam3_image_model.to(target_dtype) 
    #processed_image = processed_image.to(target_dtype) 
 
    # 4. 后续原逻辑不变 
    #inference_state = sam3_image_model.set_image(processed_image) 
    # 其他分割逻辑...

    # 勿對整個 SAM3 .to(bfloat16)：decoder/幾何等仍會產生 float32 激活，與 bf16 Linear 權重衝突。
    # 權重維持 float32，在 CUDA 上以 autocast 做混合精度（與官方 demo 一致），由 AMP 統一 matmul dtype。
    def _sam3_amp_ctx():
        if not torch.cuda.is_available() or sam3_image_model.device != "cuda":
            return contextlib.nullcontext()
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if major >= 6:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    wall_masks = []
    floor_masks = []
    for i, image in enumerate(images):
        # 1. 图像预处理
        # processed_image = processor.preprocess(image).to("cuda")
        image = Image.fromarray(image)
        # 因为使用的显卡不是Amper架构，这里精度需要对齐
        #image = processed_image.to(target_dtype)
        #image = image.bfloat16()
        #print(f"模型精度: {sam3_image_model.dtype}, 输入精度: {image.dtype}") 
        
        # In src/object_segmentation.py, before line 58
        # image = image.float()

        with _sam3_amp_ctx():
            inference_state = sam3_image_model.set_image(image)
            sam3_image_model.reset_all_prompts(inference_state)
            inference_state = sam3_image_model.set_text_prompt(
                state=inference_state, prompt="single wall"
            )
        masks = inference_state['masks'].cpu().numpy()
        for mask in masks:
            if np.sum(mask) > 500: # Filter out small masks.
                wall_masks.append({
                    'frame_id': i,
                    'mask': mask[0] # Remove the extra dimension.
                })
        with _sam3_amp_ctx():
            sam3_image_model.reset_all_prompts(inference_state)
            inference_state = sam3_image_model.set_text_prompt(
                state=inference_state, prompt="floor"
            )
        masks = inference_state['masks'].cpu().numpy()
        for mask in masks:
            if np.sum(mask) > 500: # Filter out small masks.
                floor_masks.append({
                    'frame_id': i,
                    'mask': mask[0] # Remove the extra dimension.
                })
    return wall_masks, floor_masks

def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

def segment_and_track(category, video_predictor, session_id):
    '''
    Segment raw instance masks using sam3 video tracking
    Args:
        category: A str means the category to segment
        video_predictor: the loaded sam3 video model  
        session_id: the session with loaded video frames corresponding to the video_predictor
    Returns:
        A list of list, each list represents a segmented instance and is composed
        of dicts with keys frame_id and mask (binary mask of the instance in that frame)
        [
            [
                {
                    'frame_id': int,
                    'mask': numpy array of shape (H, W) with binary values (True or False)
                },
                ...
            ],
            ...
        ]
    '''
    # Reset session and add text prompt for the category to segment
    _ = video_predictor.handle_request(request=dict(type="reset_session", session_id=session_id))
    video_predictor.handle_request(request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=category))
    outputs_per_frame = propagate_in_video(video_predictor, session_id)
    if not outputs_per_frame:
        return []

    # Collect all object IDs across frames, discontinuous segments will be split into different instances.
    all_obj_ids = set()
    for frame_idx in outputs_per_frame.keys():
        all_obj_ids.update(outputs_per_frame[frame_idx]['out_obj_ids'])
    if len(all_obj_ids) == 0:
        print(f'No object detected for {category}.')
        return []
    final_results = []
    sorted_obj_ids = sorted(list(all_obj_ids))
    for obj_id in sorted_obj_ids:
        raw_frame_ids = sorted([
            id for id in outputs_per_frame.keys() 
            if obj_id in outputs_per_frame[id]['out_obj_ids']
        ])
        
        if not raw_frame_ids:
            continue
        segments = []
        if len(raw_frame_ids) > 0:
            current_segment = [raw_frame_ids[0]]
            for i in range(1, len(raw_frame_ids)):
                if raw_frame_ids[i] == raw_frame_ids[i-1] + 1:
                    current_segment.append(raw_frame_ids[i])
                else:
                    segments.append(current_segment)
                    current_segment = [raw_frame_ids[i]]
            segments.append(current_segment)
        for frame_ids in segments:
            instance_track = []
            
            for frame_id in frame_ids:
                obj_indices = np.where(outputs_per_frame[frame_id]['out_obj_ids'] == obj_id)[0]
                
                if len(obj_indices) > 0:
                    idx = obj_indices[0]
                    raw_mask = outputs_per_frame[frame_id]['out_binary_masks'][idx].squeeze()
                    binary_mask = raw_mask > 0
                    
                    instance_track.append({
                        'frame_id': frame_id,
                        'mask': binary_mask
                    })
            if instance_track:
                final_results.append(instance_track)

    return final_results
