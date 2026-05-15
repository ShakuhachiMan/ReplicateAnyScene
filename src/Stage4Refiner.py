"""
Stage 4: Iterative Visual–Spatial Alignment（與本專案資料流相容的精簡版）

論文 Stage 4 目標：在真實視圖與渲染結果之間建立對應，迭代優化**物體位姿**（典型為
render–match–optimize）。公開程式目前以 **SAM3D 網格 + 4×4 位姿 T** 表示每個實例，
沒有論文中的可微渲染器與完整優化迴圈。

先前草稿若使用 3D Gaussian Splatting（gaussian_renderer、可微光柵、稠密化）則與本 pipeline
**不相容**：本 repo 無對應模組，且表示為 **mesh + T**，並非單一 3DGS 場景。

本模組改為 **無額外深度學習依賴** 的 **粗位姿搜尋**：
在每個實例的 **最佳視角幀** 上，將網格頂點經 T 變到世界座標，再使用與 SAM3D 前處理
一致的 **flip @ extrinsic** 投影到像素，以「投影點落在實例 mask 內的比例」為分數，
在 **繞世界 Z 軸、過物體幾何中心** 的小角度範圍內做網格搜尋，更新 T。

這能部分緩解「物體朝向與場景不一致」的問題，但**不等價**論文完整 Stage 4；若要逼近論文，
需引入可微網格／高斯渲染與多視角 photometric 損失。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

from src.pipeline_progress import log


_FLIP = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)


def _mesh_sample_points(mesh: trimesh.Trimesh, n: int, seed: int = 0) -> np.ndarray:
    """Returns (N, 3) float64 in mesh local coordinates."""
    rng = np.random.default_rng(seed)
    if mesh is None or len(mesh.vertices) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if len(mesh.vertices) <= n:
        return np.asarray(mesh.vertices, dtype=np.float64)
    idx = rng.choice(len(mesh.vertices), size=n, replace=False)
    return np.asarray(mesh.vertices[idx], dtype=np.float64)


def _scene_to_trimesh(mesh_or_scene: Any) -> Optional[trimesh.Trimesh]:
    if mesh_or_scene is None:
        return None
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    if isinstance(mesh_or_scene, trimesh.Scene):
        dumped = mesh_or_scene.dump(concatenate=True)
        if dumped is not None and len(getattr(dumped, "vertices", [])) > 0:
            return dumped
    return None


def _centroid_local(mesh: trimesh.Trimesh) -> np.ndarray:
    """Homogeneous centroid in mesh local frame."""
    c = np.asarray(mesh.bounds.mean(axis=0), dtype=np.float64)
    return np.array([c[0], c[1], c[2], 1.0], dtype=np.float64)


def _world_yaw_about_point(T: np.ndarray, centroid_world: np.ndarray, angle_rad: float) -> np.ndarray:
    """T' = M(c) Rz M(-c) @ T , world Z rotation about vertical through centroid."""
    c = np.asarray(centroid_world, dtype=np.float64).reshape(3)
    M = np.eye(4, dtype=np.float64)
    M[:3, 3] = c
    Minv = np.eye(4, dtype=np.float64)
    Minv[:3, 3] = -c
    cz, sz = np.cos(angle_rad), np.sin(angle_rad)
    Rz = np.eye(4, dtype=np.float64)
    Rz[0, 0] = cz
    Rz[0, 1] = -sz
    Rz[1, 0] = sz
    Rz[1, 1] = cz
    return (M @ Rz @ Minv @ T).astype(np.float64)


def _project_world_points(
    Xw: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Xw: (N, 3) world points
    Returns u, v, z_cam (each length N); invalid z -> nan
    """
    if Xw.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])
    N = Xw.shape[0]
    hom = np.concatenate([Xw, np.ones((N, 1), dtype=np.float64)], axis=1)
    Xc = (_FLIP @ extrinsic @ hom.T).T[:, :3]
    z = Xc[:, 2]
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    u = fx * (Xc[:, 0] / np.maximum(z, 1e-8)) + cx
    v = fy * (Xc[:, 1] / np.maximum(z, 1e-8)) + cy
    bad = z <= 1e-6
    u = u.astype(np.float64)
    v = v.astype(np.float64)
    u[bad] = np.nan
    v[bad] = np.nan
    return u, v, z


def _mask_inlier_score(
    u: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray,
    *,
    ref_image_hw: Optional[Tuple[int, int]] = None,
) -> float:
    """
    計算投影點落在 mask 內的比例。
    若 ref_image_hw=(Hc,Wc) 與 mask (Hm,Wm) 不同，將 (u,v) 視為在「內參/投影所對應的參考影像」
    座標系下，先線性縮放到 mask 解析度再取樣（常見於 VGGT 內參與實際輸出圖尺寸不完全一致）。
    最後對索引做 clip，避免越界。
    """
    if mask is None or mask.size == 0:
        return 0.0
    Hm, Wm = int(mask.shape[0]), int(mask.shape[1])
    u = np.asarray(u, dtype=np.float64).copy()
    v = np.asarray(v, dtype=np.float64).copy()
    if ref_image_hw is not None:
        Hc, Wc = int(ref_image_hw[0]), int(ref_image_hw[1])
        if Hc > 0 and Wc > 0 and (Hc != Hm or Wc != Wm):
            u = u * (Wm / float(Wc))
            v = v * (Hm / float(Hc))
    valid = np.isfinite(u) & np.isfinite(v)
    if not np.any(valid):
        return 0.0
    ui = np.round(u[valid]).astype(np.int32)
    vi = np.round(v[valid]).astype(np.int32)
    ui = np.clip(ui, 0, Wm - 1)
    vi = np.clip(vi, 0, Hm - 1)
    inside = mask.astype(bool)[vi, ui]
    return float(np.mean(inside))


class Stage4Refiner:
    """
    以最佳視角 mask 為參考的繞世界 Z 軸 yaw 粗搜尋（可視作 Stage 4 的極簡占位實作）。
    """

    def __init__(
        self,
        n_vertices_sample: int = 2048,
        yaw_range_deg: float = 40.0,
        n_yaw: int = 17,
        seed: int = 0,
    ):
        self.n_vertices_sample = int(n_vertices_sample)
        self.yaw_range_deg = float(yaw_range_deg)
        self.n_yaw = int(max(3, n_yaw))
        self.seed = int(seed)

    def refine_instance_pose(
        self,
        instance_info: Dict[str, Any],
        mask: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        frame_id: int = 0,
        *,
        ref_image_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        mesh = _scene_to_trimesh(instance_info.get("original_mesh"))
        if mesh is None:
            log(f"[Stage4] skip instance (no mesh) frame={frame_id}")
            return instance_info
        if mask is None or not np.any(mask):
            log(f"[Stage4] skip instance (empty mask) frame={frame_id}")
            return instance_info

        if ref_image_hw is not None:
            Hc, Wc = int(ref_image_hw[0]), int(ref_image_hw[1])
            Hm, Wm = int(mask.shape[0]), int(mask.shape[1])
            if Hc != Hm or Wc != Wm:
                log(
                    f"[Stage4] frame={frame_id} mask {Hm}x{Wm} vs ref image {Hc}x{Wc}: "
                    f"will scale projected (u,v) to mask resolution"
                )

        T0 = np.asarray(instance_info["T"], dtype=np.float64)
        if T0.shape != (4, 4):
            return instance_info

        pts_local = _mesh_sample_points(mesh, self.n_vertices_sample, seed=self.seed + frame_id)
        if pts_local.shape[0] == 0:
            return instance_info

        c_local = _centroid_local(mesh)
        c_world = (T0 @ c_local)[:3]

        def _score_for_T(T_try: np.ndarray) -> float:
            hom = np.concatenate(
                [pts_local, np.ones((pts_local.shape[0], 1), dtype=np.float64)], axis=1
            )
            Xw = (T_try @ hom.T).T[:, :3]
            u, v, _ = _project_world_points(Xw, extrinsic, intrinsic)
            return _mask_inlier_score(u, v, mask.astype(bool), ref_image_hw=ref_image_hw)

        baseline = _score_for_T(T0)
        best_T = T0.copy()
        best_score = baseline
        yaws = np.linspace(
            np.radians(-self.yaw_range_deg),
            np.radians(self.yaw_range_deg),
            self.n_yaw,
        )
        for yaw in yaws:
            T_try = _world_yaw_about_point(T0, c_world, yaw)
            score = _score_for_T(T_try)
            if score > best_score:
                best_score = score
                best_T = T_try

        if best_score > baseline + 1e-4:
            instance_info["T"] = best_T.astype(np.float32)
            log(
                f"[Stage4] frame={frame_id} mask_inlier {baseline:.3f} -> {best_score:.3f} "
                f"(yaw ±{self.yaw_range_deg:.1f}°, {self.n_yaw} steps)"
            )
        else:
            log(
                f"[Stage4] frame={frame_id} no pose change (best={best_score:.3f}, baseline={baseline:.3f})"
            )
        return instance_info


def _mask_for_instance_at_frame(
    instance_masks: List[Dict[str, Any]],
    frame_id: int,
    shape_hw: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """取該幀第一個符合的 mask；若給定 shape_hw 則優先匹配該解析度。"""
    best: Optional[np.ndarray] = None
    for item in instance_masks:
        if int(item["frame_id"]) != int(frame_id):
            continue
        m = np.asarray(item["mask"], dtype=bool)
        if m.ndim < 2:
            continue
        if shape_hw is not None and m.shape[:2] == shape_hw:
            return m
        if best is None:
            best = m
    return best


def run_stage4_on_instances(
    all_instances: Dict[str, List[Dict[str, Any]]],
    all_optimal_frame_ids: Dict[str, List[int]],
    deduplicated_all_masks: Dict[str, List[List[Dict[str, Any]]]],
    colors: np.ndarray,
    extrinsics: np.ndarray,
    intrinsic: np.ndarray,
    *,
    n_vertices_sample: int = 2048,
    yaw_range_deg: float = 40.0,
    n_yaw: int = 17,
    seed: int = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    In-place 更新 all_instances[*][*]['T']。需要與 all_optimal_frame_ids、deduplicated_all_masks 對齊。
    """
    refiner = Stage4Refiner(
        n_vertices_sample=n_vertices_sample,
        yaw_range_deg=yaw_range_deg,
        n_yaw=n_yaw,
        seed=seed,
    )
    H, W = int(colors.shape[1]), int(colors.shape[2])
    for category, instances in all_instances.items():
        if category not in all_optimal_frame_ids or category not in deduplicated_all_masks:
            continue
        fids = all_optimal_frame_ids[category]
        masks_per_inst = deduplicated_all_masks[category]
        for i, inst in enumerate(instances):
            if i >= len(fids) or i >= len(masks_per_inst):
                continue
            fid = int(fids[i])
            if fid < 0 or fid >= extrinsics.shape[0]:
                continue
            mask = _mask_for_instance_at_frame(masks_per_inst[i], fid, (H, W))
            if mask is None or not np.any(mask):
                log(f"[Stage4] skip {category}[{i}] (no mask at optimal frame {fid})")
                continue
            ext = np.asarray(extrinsics[fid], dtype=np.float64)
            if ext.shape == (3, 4):
                e4 = np.eye(4, dtype=np.float64)
                e4[:3, :] = ext
                ext = e4
            ref_hw = (H, W)
            refiner.refine_instance_pose(
                inst,
                mask,
                ext,
                np.asarray(intrinsic, dtype=np.float64),
                frame_id=fid,
                ref_image_hw=ref_hw,
            )
    return all_instances
