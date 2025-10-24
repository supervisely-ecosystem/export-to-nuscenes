import math
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import supervisely as sly
from nuscenes.nuscenes import NuScenes

# -------------------------
# Config / conventions
# -------------------------
tmp_dir = "./tmp"
NUSCENES_VER = "v1.0-mini"
EXTRINSIC_CONVENTION = "world_to_sensor"  # or "sensor_to_world"
CAM_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
PC_EXTS = {".pcd", ".ply", ".las", ".bin"}  # keep real ext as is
LIDAR_CHANNEL = "LIDAR_TOP"
project_type = None


class WorkingProjectType:
    POINT_CLOUD_EPISODE = "point_cloud_episode"
    POINT_CLOUD = "point_cloud"


# -------------------------
# Math helpers
# -------------------------
def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> list[float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


def _mat34_to_rt(E34):
    E34 = np.asarray(E34, dtype=float).reshape(3, 4)
    R, t = E34[:, :3], E34[:, 3]
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R, t


def _invert_rt(R, t):
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def _compose_rt(Ra, ta, Rb, tb):
    R = Ra @ Rb
    t = Ra @ tb + ta
    return R, t


def _rot_to_quat_wxyz(R):
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    return [float(w), float(x), float(y), float(z)]


def _get_episode_like_annotation_structure(api: sly.Api, dataset_id: int) -> Dict[str, Any]:
    ann = {"datasetId": dataset_id, "frames": []}
    objects_all = []
    for i, pcd in enumerate(api.pointcloud.get_list(dataset_id)):
        n = api.pointcloud.annotation.download(pcd.id)
        for object in n["objects"]:
            if object not in objects_all:
                objects_all.append(object)
        ann["frames"].append({"index": i, "pointCloudId": pcd.id, "figures": n.get("figures", [])})
    ann["objects"] = objects_all
    return ann


def build_nuscenes_records(
    extrinsic_3x4,
    intrinsic_3x3,
    timestamp_us,
    extrinsic_convention=EXTRINSIC_CONVENTION,
    rig_R_se=None,
    rig_t_se=None,
):
    R_e, t_e = _mat34_to_rt(extrinsic_3x4)
    if extrinsic_convention == "world_to_sensor":
        R_sw, t_sw = _invert_rt(R_e, t_e)
    elif extrinsic_convention == "sensor_to_world":
        R_sw, t_sw = R_e, t_e
    else:
        raise ValueError("extrinsic_convention must be 'world_to_sensor' or 'sensor_to_world'")

    if rig_R_se is None:
        rig_R_se = np.eye(3)
    if rig_t_se is None:
        rig_t_se = np.zeros(3)

    R_se_inv, t_se_inv = _invert_rt(rig_R_se, rig_t_se)
    R_ew, t_ew = _compose_rt(R_sw, t_sw, R_se_inv, t_se_inv)

    ego_pose = {
        "token": uuid.uuid4().hex,
        "timestamp": int(timestamp_us),
        "translation": [float(t_ew[0]), float(t_ew[1]), float(t_ew[2])],
        "rotation": _rot_to_quat_wxyz(R_ew),
    }
    calibrated_sensor = {
        "token": uuid.uuid4().hex,
        "sensor_token": None,  # set later
        "translation": [float(rig_t_se[0]), float(rig_t_se[1]), float(rig_t_se[2])],
        "rotation": _rot_to_quat_wxyz(rig_R_se),
    }
    if intrinsic_3x3 is not None:
        calibrated_sensor["camera_intrinsic"] = (
            np.asarray(intrinsic_3x3, float).reshape(3, 3).tolist()
        )

    return ego_pose, calibrated_sensor


# -------------------------
# IO helpers
# -------------------------
def _new_token() -> str:
    return uuid.uuid4().hex


def _write_json(path: Path, data: Any):
    sly.json.dump_json_file(data, path)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _find_sibling_image(json_path: Path) -> Optional[Path]:
    stem = json_path.stem
    parent = json_path.parent
    for ext in CAM_EXTS:
        cand = parent / f"{stem}{ext}"
        if cand.exists():
            return cand
    for item in parent.iterdir():
        if item.is_file() and item.suffix.lower() in CAM_EXTS and item.stem.startswith(stem):
            return item
    return None


def _safe_int(v, default=0) -> int:
    try:
        return int(v)
    except Exception:
        return default


# -------------------------
# Project parsing
# -------------------------
def _parse_rimages_for_sensors(rimage_folder: Path) -> set[str]:
    jsons_paths = sly.fs.list_files_recursively(
        rimage_folder.as_posix(), [".json"], ignore_valid_extensions_case=True
    )
    sensor_names = set()
    for json_path in jsons_paths:
        d = sly.json.load_json_file(json_path)
        sensor_name = d.get("meta", {}).get("deviceId", None)
        if sensor_name:
            sensor_names.add(sensor_name)
    return sensor_names


def _load_frame_pcd_map(dataset_folder: Path) -> Dict[int, Dict[str, Any]]:
    """
    Tries to read dataset_folder/frame_pointcloud_map.json and returns
    {frame_idx: {"pointcloud_path": str, "pointcloud_id": int (optional), "timestamp": int (optional)}}
    """
    result: Dict[int, Dict[str, Any]] = {}
    cand = dataset_folder / "frame_pointcloud_map.json"
    if not cand.exists():
        return result
    data = sly.json.load_json_file(cand)
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                fi = int(v.get("frame", k))
            except Exception:
                continue
            path = (
                v.get("path")
                or v.get("pointcloud_path")
                or v.get("filePath")
                or v.get("file")
                or v.get("name")
            )
            pid = v.get("pointCloudId") or v.get("id")
            ts = v.get("timestamp")
            if path:
                result[fi] = {"pointcloud_path": str(path), "pointcloud_id": pid, "timestamp": ts}
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            fi = item.get("frame")
            if fi is None:
                continue
            path = (
                item.get("path")
                or item.get("pointcloud_path")
                or item.get("filePath")
                or item.get("file")
                or item.get("name")
            )
            pid = item.get("pointCloudId") or item.get("id")
            ts = item.get("timestamp")
            if path:
                result[int(fi)] = {
                    "pointcloud_path": str(path),
                    "pointcloud_id": pid,
                    "timestamp": ts,
                }
    return result


def _collect_related_images(dataset_folder: Path) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """
    Returns mapping: frame_idx -> { sensor_name -> {"json": Path, "image": Path, "meta": dict, "sensorsData": dict} }
    Uses meta.frame if available; otherwise skips those entries.
    """
    rimage_folder = dataset_folder / "related_images"
    out: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    if not rimage_folder.exists():
        return out
    jsons = sly.fs.list_files_recursively(
        rimage_folder.as_posix(), [".json"], ignore_valid_extensions_case=True
    )
    for jp in jsons:
        jp = Path(jp)
        d = sly.json.load_json_file(jp)
        meta = d.get("meta", {})
        sensors_data = d.get("sensorsData", {})
        sensor_name = meta.get("deviceId")
        frame_idx = meta.get("frame")
        if sensor_name is None or frame_idx is None:
            continue  # require frame index for reliable wiring
        img_path = _find_sibling_image(jp)
        out[int(frame_idx)][sensor_name] = {
            "json": jp,
            "image": img_path,
            "meta": meta,
            "sensorsData": sensors_data,
        }
    return out


# -------------------------
# nuScenes builders
# -------------------------
def _build_taxonomy(meta_json_path: Path, ann_path: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    meta = sly.json.load_json_file(meta_json_path)
    class2token: Dict[int, str] = {}
    tag2token: Dict[int, str] = {}

    categories = []
    for objclass in meta["classes"]:
        title = objclass["title"]
        desc = objclass.get("description", "")
        cid = objclass["id"]
        tok = _new_token()
        class2token[cid] = tok
        categories.append({"token": tok, "name": title, "description": desc})
    _write_json(ann_path / "category.json", categories)

    attributes = []
    for attr in meta.get("tags", []):
        name = attr["name"]
        tok = _new_token()
        tid = attr["id"]
        tag2token[tid] = tok
        attributes.append({"token": tok, "name": name, "description": ""})
    _write_json(ann_path / "attribute.json", attributes)

    # Visibility: standard 4 levels
    visibility = [
        {"token": "1", "description": "visibility 0-40%"},
        {"token": "2", "description": "visibility 40-60%"},
        {"token": "3", "description": "visibility 60-80%"},
        {"token": "4", "description": "visibility 80-100%"},
    ]
    _write_json(ann_path / "visibility.json", visibility)

    _write_json(ann_path / "map.json", [])
    return class2token, tag2token


def _collect_sensors(
    local_project_path: Path, add_lidar: bool = True
) -> Tuple[List[dict], Dict[str, str]]:
    sensor2token: Dict[str, str] = {}
    sensors: List[dict] = []
    for dataset_folder in local_project_path.iterdir():
        if not dataset_folder.is_dir():
            continue
        rimage_folder = dataset_folder / "related_images"
        sensor_names = _parse_rimages_for_sensors(rimage_folder)
        for sensor_name in sorted(sensor_names):
            if sensor_name in sensor2token:
                continue
            tok = _new_token()
            sensor2token[sensor_name] = tok
            sensors.append({"token": tok, "channel": sensor_name, "modality": "camera"})
    if add_lidar:
        if LIDAR_CHANNEL not in sensor2token:
            tok = _new_token()
            sensor2token[LIDAR_CHANNEL] = tok
            sensors.append({"token": tok, "channel": LIDAR_CHANNEL, "modality": "lidar"})
    return sensors, sensor2token


def _collect_calibrated_sensors(
    local_project_path: Path, sensor2token: Dict[str, str]
) -> Tuple[List[dict], Dict[str, str]]:
    """
    Build one calibrated_sensor per channel. T_sensor_ego defaults to identity; if a camera's
    sensorsData.extrinsicMatrix looks like a static rig (small translation norm), use it.
    camera_intrinsic from JSON if available.
    """
    calib_by_sensor: Dict[str, dict] = {}
    cal_token_by_sensor: Dict[str, str] = {}

    for dataset_folder in local_project_path.iterdir():
        if not dataset_folder.is_dir():
            continue
        rim_by_frame = _collect_related_images(dataset_folder)
        for frame_map in rim_by_frame.values():
            for sensor_name, info in frame_map.items():
                if sensor_name in calib_by_sensor:
                    continue
                intrinsic = info.get("sensorsData", {}).get("intrinsicMatrix")
                extrinsic = info.get("sensorsData", {}).get("extrinsicMatrix")
                cal = {
                    "token": _new_token(),
                    "sensor_token": sensor2token[sensor_name],
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                }
                if extrinsic is not None:
                    try:
                        R_e, t_e = _mat34_to_rt(extrinsic)
                        if float(np.linalg.norm(t_e)) < 5.0:  # treat as static rig
                            cal["translation"] = [float(t_e[0]), float(t_e[1]), float(t_e[2])]
                            cal["rotation"] = _rot_to_quat_wxyz(R_e)
                    except Exception:
                        pass
                if intrinsic is not None:
                    cal["camera_intrinsic"] = np.asarray(intrinsic, float).reshape(3, 3).tolist()
                calib_by_sensor[sensor_name] = cal
                cal_token_by_sensor[sensor_name] = cal["token"]

    if LIDAR_CHANNEL in sensor2token and LIDAR_CHANNEL not in calib_by_sensor:
        cal = {
            "token": _new_token(),
            "sensor_token": sensor2token[LIDAR_CHANNEL],
            "translation": [0.0, 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
        }
        calib_by_sensor[LIDAR_CHANNEL] = cal
        cal_token_by_sensor[LIDAR_CHANNEL] = cal["token"]

    return list(calib_by_sensor.values()), cal_token_by_sensor


def _compute_ego_pose_from_any_cam(info: dict) -> Optional[dict]:
    sensors_data = info.get("sensorsData", {})
    meta = info.get("meta", {})
    extrinsic = sensors_data.get("extrinsicMatrix")
    intrinsic = sensors_data.get("intrinsicMatrix")
    timestamp = meta.get("timestamp")
    if extrinsic is None or timestamp is None:
        return None
    try:
        _, t = _mat34_to_rt(extrinsic)
        if float(np.linalg.norm(t)) < 5.0:  # small rig offset
            return {
                "token": _new_token(),
                "timestamp": int(timestamp),
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            }
    except Exception:
        pass
    ego, _ = build_nuscenes_records(
        extrinsic_3x4=extrinsic,
        intrinsic_3x3=intrinsic,
        timestamp_us=timestamp,
        extrinsic_convention=EXTRINSIC_CONVENTION,
        rig_R_se=None,
        rig_t_se=None,
    )
    return ego


def _copy_into_samples(src: Path, samples_root: Path, channel: str) -> Tuple[str, str]:
    """
    Copies src into samples/{channel}/ and returns (rel_path, fileformat)
    """
    target_dir = samples_root / channel
    _ensure_dir(target_dir)
    dst = target_dir / src.name
    shutil.copy2(src, dst)
    rel = str(dst.relative_to(samples_root.parent))  # relative to dataroot
    fileformat = src.suffix.lstrip(".").lower()
    return rel, fileformat


def _to_nuscenes_sample_ann(
    box: dict,
    *,
    category_name: str,
    sample_token: str,
    instance_token: Optional[str] = None,
    attribute_tokens: Optional[List[str]] = None,
    visibility_token: str = "1",
    num_lidar_pts: int = 0,
    num_radar_pts: int = 0,
    prev: str = "",
    next_: str = "",
    angles_in_degrees: bool = False,
    dims_map: Tuple[str, str, str] = ("x", "y", "z"),
) -> dict:
    pos = box.get("position", {})
    rot = box.get("rotation", {})
    dims = box.get("dimensions", {})

    translation = [
        float(pos.get("x", 0.0)),
        float(pos.get("y", 0.0)),
        float(pos.get("z", 0.0)),
    ]
    w_key, l_key, h_key = dims_map
    size = [
        float(dims.get(w_key, 0.0)),
        float(dims.get(l_key, 0.0)),
        float(dims.get(h_key, 0.0)),
    ]
    roll = float(rot.get("x", 0.0))
    pitch = float(rot.get("y", 0.0))
    yaw = float(rot.get("z", 0.0))
    if angles_in_degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
    quaternion = _euler_to_quaternion(roll, pitch, yaw)

    token = _new_token()
    if instance_token is None:
        instance_token = _new_token()
    if attribute_tokens is None:
        attribute_tokens = []

    return {
        "attribute_tokens": attribute_tokens,
        "category_name": category_name,
        "instance_token": instance_token,
        "next": next_,
        "num_lidar_pts": int(num_lidar_pts),
        "num_radar_pts": int(num_radar_pts),
        "prev": prev,
        "rotation": quaternion,
        "sample_token": sample_token,
        "size": size,
        "token": token,
        "translation": translation,
        "visibility_token": visibility_token,
    }


def _build_annotations_and_instances(
    api: sly.Api,
    dataset_id: int,
    class_id_to_token: Dict[int, str],
    sample_token_by_pcd_id: Dict[int, str],
) -> Tuple[List[dict], List[dict], Dict[str, List[str]]]:
    global project_type
    if project_type == WorkingProjectType.POINT_CLOUD:
        ann = _get_episode_like_annotation_structure(api, dataset_id)
    else:
        ann = api.pointcloud_episode.annotation.download(dataset_id)

    anns: List[dict] = []
    object_id_to_instance_token: Dict[int, str] = {}
    inst_to_anns: DefaultDict[str, List[dict]] = defaultdict(list)
    instances_rows: Dict[str, dict] = {}
    object_id_to_name = {}
    sample_to_ann_tokens: DefaultDict[str, List[str]] = defaultdict(list)

    for obj in ann["objects"]:
        object_id = obj["id"]
        object_id_to_name[object_id] = obj["classTitle"]
        instance_token = _new_token()
        class_token = class_id_to_token.get(obj["classId"])
        instances_rows[instance_token] = {
            "category_token": class_token,
            "first_annotation_token": "",
            "last_annotation_token": "",
            "nbr_annotations": 0,
            "token": instance_token,
        }
        object_id_to_instance_token[object_id] = instance_token

    for frame in ann.get("frames", [ann]):
        pcd_id = frame["pointCloudId"]
        sample_token = sample_token_by_pcd_id.get(pcd_id)
        if sample_token is None:
            continue
        for figure in frame.get("figures", []):
            box = figure["geometry"]
            category_name = object_id_to_name.get(figure["objectId"], "unknown")
            object_id = figure["objectId"]
            instance_token = object_id_to_instance_token.get(object_id)
            nuscenes_ann = _to_nuscenes_sample_ann(
                box,
                category_name=category_name,
                sample_token=sample_token,
                instance_token=instance_token,
                attribute_tokens=[],
                visibility_token="1",
            )
            anns.append(nuscenes_ann)
            inst_to_anns[instance_token].append(nuscenes_ann)
            sample_to_ann_tokens[sample_token].append(nuscenes_ann["token"])

    for instance_token, items in inst_to_anns.items():
        for i, it in enumerate(items):
            if i > 0:
                it["prev"] = items[i - 1]["token"]
            if i < len(items) - 1:
                it["next"] = items[i + 1]["token"]
        if items:
            instances_rows[instance_token]["first_annotation_token"] = items[0]["token"]
            instances_rows[instance_token]["last_annotation_token"] = items[-1]["token"]
            instances_rows[instance_token]["nbr_annotations"] = len(items)

    instances = list(instances_rows.values())
    return anns, instances, dict(sample_to_ann_tokens)


# -------------------------
# Main conversion
# -------------------------
def convert_sly_project_to_nuscenes(api: sly.Api, project_id, dest_dir):
    global project_type
    project_info = api.project.get_info_by_id(project_id)
    if project_info.type == sly.ProjectType.POINT_CLOUD_EPISODES.value:
        project_type = WorkingProjectType.POINT_CLOUD_EPISODE
    elif project_info.type == sly.ProjectType.POINT_CLOUDS.value:
        project_type = WorkingProjectType.POINT_CLOUD
    else:
        raise ValueError(
            "Project type not supported. Please provide a Point Cloud or Point Cloud Episode project."
        )

    local_project_path = Path(tmp_dir) / project_info.name
    if not sly.fs.dir_exists(local_project_path.as_posix()):
        if project_type == WorkingProjectType.POINT_CLOUD_EPISODE:
            from supervisely.project.pointcloud_episode_project import (
                download_pointcloud_episode_project,
            )

            download_pointcloud_episode_project(api, project_id, local_project_path.as_posix())
        elif project_type == WorkingProjectType.POINT_CLOUD:
            from supervisely.project.pointcloud_project import download_pointcloud_project

            download_pointcloud_project(api, project_id, local_project_path.as_posix())

    dest_dir = Path(dest_dir)
    maps_path = dest_dir / "maps"
    samples_path = dest_dir / "samples"
    sweeps_path = dest_dir / "sweeps"
    ann_path = dest_dir / NUSCENES_VER
    _ensure_dir(maps_path)
    _ensure_dir(samples_path)
    _ensure_dir(sweeps_path)
    _ensure_dir(ann_path)

    class2token, tag2token = _build_taxonomy(local_project_path / "meta.json", ann_path)

    sensors, sensor2token = _collect_sensors(local_project_path)
    _write_json(ann_path / "sensor.json", sensors)

    calibrated_sensors, cal_token_by_sensor = _collect_calibrated_sensors(
        local_project_path, sensor2token
    )
    _write_json(ann_path / "calibrated_sensor.json", calibrated_sensors)
    missing_channels = [ch for ch in sensor2token.keys() if ch not in cal_token_by_sensor]
    if missing_channels:
        for ch in missing_channels:
            cal = {
                "token": _new_token(),
                "sensor_token": sensor2token[ch],
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            }
            calibrated_sensors.append(cal)
            cal_token_by_sensor[ch] = cal["token"]
        _write_json(ann_path / "calibrated_sensor.json", calibrated_sensors)

    def _ensure_cal(channel: str) -> str:
        tok = cal_token_by_sensor.get(channel)
        if tok is not None:
            return tok
        cal = {
            "token": _new_token(),
            "sensor_token": sensor2token[channel],
            "translation": [0.0, 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
        }
        calibrated_sensors.append(cal)
        cal_token_by_sensor[channel] = cal["token"]
        return cal["token"]

    logs: List[dict] = []
    log_token_by_dataset: Dict[str, str] = {}

    scenes: List[dict] = []
    samples: List[dict] = []
    sample_data: List[dict] = []
    ego_poses: List[dict] = []

    last_sd_token_per_channel: Dict[str, Optional[str]] = defaultdict(lambda: None)

    sample_token_by_pcd_id_global: Dict[int, str] = {}
    sample_anns_map_global: Dict[str, List[str]] = {}

    for dataset in api.dataset.get_list(project_id):
        dataset_folder = local_project_path / dataset.name
        if not dataset_folder.exists():
            continue

        log_token = _new_token()
        log_token_by_dataset[dataset.name] = log_token
        log_row = {
            "token": log_token,
            "logfile": f"{dataset.name}.log",
            "vehicle": "n/a",
            "date_captured": "n/a",
            "location": "n/a",
        }
        logs.append(log_row)

        rimgs_by_frame = _collect_related_images(dataset_folder)
        frame_pcd_map = _load_frame_pcd_map(dataset_folder)

        if project_type == WorkingProjectType.POINT_CLOUD:
            ann = _get_episode_like_annotation_structure(api, dataset.id)
        else:
            ann = api.pointcloud_episode.annotation.download(dataset.id)

        frames = list(ann.get("frames", []))
        frames.sort(key=lambda f: f.get("index", 0))

        scene_token = _new_token()
        scene_name = dataset.name
        sample_tokens_order: List[str] = []

        prev_sample_token: Optional[str] = None

        for frame in frames:
            frame_idx = frame.get("index", 0)
            pcd_id = frame.get("pointCloudId")
            sample_token = _new_token()
            sample_tokens_order.append(sample_token)
            sample_token_by_pcd_id_global[pcd_id] = sample_token

            ts = 0
            if frame_idx in rimgs_by_frame:
                any_cam = next(iter(rimgs_by_frame[frame_idx].values()))
                ts = _safe_int(any_cam.get("meta", {}).get("timestamp", 0))
            if ts == 0 and frame_idx in frame_pcd_map:
                ts = _safe_int(frame_pcd_map[frame_idx].get("timestamp", 0))

            sample_row = {
                "token": sample_token,
                "timestamp": ts,
                "scene_token": scene_token,
                "next": "",
                "prev": prev_sample_token or "",
                "data": {},
                "anns": [],
            }
            if prev_sample_token:
                for s in samples[::-1]:
                    if s["token"] == prev_sample_token:
                        s["next"] = sample_token
                        break
            samples.append(sample_row)
            prev_sample_token = sample_token
            current_sample_row = sample_row

            ego_row = None
            if frame_idx in rimgs_by_frame and len(rimgs_by_frame[frame_idx]) > 0:
                for info in rimgs_by_frame[frame_idx].values():
                    ego_row = _compute_ego_pose_from_any_cam(info)
                    if ego_row:
                        break
            if ego_row is None:
                ego_row = {
                    "token": _new_token(),
                    "timestamp": ts,
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                }
            ego_poses.append(ego_row)
            ego_pose_token = ego_row["token"]

            pcd_local: Optional[Path] = None
            if frame_idx in frame_pcd_map and frame_pcd_map[frame_idx].get("pointcloud_path"):
                pth = frame_pcd_map[frame_idx]["pointcloud_path"]
                cand = dataset_folder / pth
                if cand.exists():
                    pcd_local = cand
            if pcd_local is None:
                pcands = [
                    p
                    for p in dataset_folder.rglob("*")
                    if p.is_file() and p.suffix.lower() in PC_EXTS
                ]
                for p in sorted(pcands):
                    if str(frame_idx) in p.stem:
                        pcd_local = p
                        break
                if pcd_local is None and pcands:
                    pcd_local = sorted(pcands)[min(frame_idx, len(pcands) - 1)]

            if pcd_local is not None:
                rel_path, fileformat = _copy_into_samples(pcd_local, samples_path, LIDAR_CHANNEL)
                sd_token = _new_token()
                sd = {
                    "token": sd_token,
                    "sample_token": sample_token,
                    "ego_pose_token": ego_pose_token,
                    "calibrated_sensor_token": _ensure_cal(LIDAR_CHANNEL),
                    "timestamp": ts,
                    "fileformat": pcd_local.suffix.lstrip(".").lower(),
                    "is_key_frame": True,
                    "height": 0,
                    "width": 0,
                    "filename": rel_path,
                    "sensor_modality": "lidar",
                    "channel": LIDAR_CHANNEL,
                    "prev": last_sd_token_per_channel[LIDAR_CHANNEL] or "",
                    "next": "",
                }
                if last_sd_token_per_channel[LIDAR_CHANNEL]:
                    for sd_prev in reversed(sample_data):
                        if sd_prev["token"] == last_sd_token_per_channel[LIDAR_CHANNEL]:
                            sd_prev["next"] = sd_token
                            break
                last_sd_token_per_channel[LIDAR_CHANNEL] = sd_token
                sample_data.append(sd)
                current_sample_row["data"][LIDAR_CHANNEL] = sd_token

            if frame_idx in rimgs_by_frame:
                for sensor_name, info in rimgs_by_frame[frame_idx].items():
                    img_path = info.get("image")
                    if img_path is None or not img_path.exists():
                        continue
                    rel_path, fileformat = _copy_into_samples(img_path, samples_path, sensor_name)
                    height = int(info.get("meta", {}).get("height", 0))
                    width = int(info.get("meta", {}).get("width", 0))
                    ts_cam = _safe_int(info.get("meta", {}).get("timestamp", ts))
                    sd_token = _new_token()
                    sd = {
                        "token": sd_token,
                        "sample_token": sample_token,
                        "ego_pose_token": ego_pose_token,
                        "calibrated_sensor_token": _ensure_cal(sensor_name),
                        "timestamp": ts_cam,
                        "fileformat": fileformat,
                        "is_key_frame": True,
                        "height": height,
                        "width": width,
                        "filename": rel_path,
                        "sensor_modality": "camera",
                        "channel": sensor_name,
                        "prev": last_sd_token_per_channel[sensor_name] or "",
                        "next": "",
                    }
                    if last_sd_token_per_channel[sensor_name]:
                        for sd_prev in reversed(sample_data):
                            if sd_prev["token"] == last_sd_token_per_channel[sensor_name]:
                                sd_prev["next"] = sd_token
                                break
                    last_sd_token_per_channel[sensor_name] = sd_token
                    sample_data.append(sd)
                    current_sample_row["data"][sensor_name] = sd_token

            for channel in sensor2token.keys():
                if channel in current_sample_row["data"]:
                    continue
                sd_token = _new_token()
                modality = "lidar" if channel == LIDAR_CHANNEL else "camera"
                sd_stub = {
                    "token": sd_token,
                    "sample_token": sample_token,
                    "ego_pose_token": ego_pose_token,
                    "calibrated_sensor_token": _ensure_cal(channel),
                    "timestamp": ts,
                    "fileformat": "",
                    "is_key_frame": False,
                    "height": 0,
                    "width": 0,
                    "filename": "",
                    "sensor_modality": modality,
                    "channel": channel,
                    "prev": last_sd_token_per_channel[channel] or "",
                    "next": "",
                }
                if last_sd_token_per_channel[channel]:
                    for sd_prev in reversed(sample_data):
                        if sd_prev["token"] == last_sd_token_per_channel[channel]:
                            sd_prev["next"] = sd_token
                            break
                last_sd_token_per_channel[channel] = sd_token
                sample_data.append(sd_stub)
                current_sample_row["data"][channel] = sd_token

        scene_row = {
            "token": scene_token,
            "name": scene_name,
            "description": "",
            "log_token": log_token,
            "first_sample_token": sample_tokens_order[0] if sample_tokens_order else "",
            "last_sample_token": sample_tokens_order[-1] if sample_tokens_order else "",
            "nbr_samples": len(sample_tokens_order),
        }
        scenes.append(scene_row)

    all_anns: List[dict] = []
    all_instances: List[dict] = []
    for dataset in api.dataset.get_list(project_id):
        anns, instances, sample_to_anns = _build_annotations_and_instances(
            api,
            dataset.id,
            class_id_to_token=class2token,
            sample_token_by_pcd_id=sample_token_by_pcd_id_global,
        )
        all_anns.extend(anns)
        all_instances.extend(instances)
        for st, toks in sample_to_anns.items():
            if st not in sample_anns_map_global:
                sample_anns_map_global[st] = []
            sample_anns_map_global[st].extend(toks)

    # backfill per-sample anns list
    for s in samples:
        s["anns"] = sample_anns_map_global.get(s["token"], [])

    maps_table = (
        [
            {
                "token": _new_token(),
                "category": "semantic_prior",
                "filename": "",  # no semantic prior available
                "log_tokens": [row["token"] for row in logs],
            }
        ]
        if len(logs) > 0
        else [
            {
                "token": _new_token(),
                "category": "semantic_prior",
                "filename": "",
                "log_tokens": [],
            }
        ]
    )

    _write_json(ann_path / "log.json", logs)
    _write_json(ann_path / "scene.json", scenes)
    _write_json(ann_path / "sample.json", samples)
    _write_json(ann_path / "ego_pose.json", ego_poses)
    _write_json(ann_path / "calibrated_sensor.json", calibrated_sensors)
    _write_json(ann_path / "sample_data.json", sample_data)
    _write_json(ann_path / "sample_annotation.json", all_anns)
    _write_json(ann_path / "instance.json", all_instances)
    _write_json(ann_path / "map.json", maps_table)

    try:
        ns = NuScenes(version=NUSCENES_VER, dataroot=dest_dir.as_posix(), verbose=True)
    except Exception as e:
        sly.logger.error(f"NuScenes failed validation due to error: {repr(e)}")
