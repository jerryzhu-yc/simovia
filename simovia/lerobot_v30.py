"""LeRobot v2.1 -> v3.0 就地升级（lerobot >= 0.6 只接受 v3.0）。

为什么不用官方 convert_dataset_v21_to_v30：它假设 v2.1 的相机是"抽帧成
图片"（dtype=image、逐帧 PNG），对"真视频版 v2.1"（dtype=video、每集一个
mp4）有两个硬缺陷：

  1. get_image_keys 只认 dtype == "image"，于是 data parquet 里残留的
     video struct 列不会被丢掉，v3.0 loader 按 7 列纯数值 schema 去 cast
     时直接 `column names don't match`；
  2. 视频特征的 names 被写成 null，policy 侧 dataset_to_policy_features
     读 names[2] 判通道维位置，会 TypeError: 'NoneType' is not subscriptable。

本模块按 v3.0 落盘规范直接重写，只做数据搬运，不重新编码画面：
    data/chunk-000/file-000.parquet        所有集的标量列拼接
    videos/<cam>/chunk-000/file-000.mp4    各路相机按集顺序无损拼接
    meta/episodes/chunk-000/file-000.parquet  每集一行（含视频时间戳与统计）
    meta/{info.json,stats.json,tasks.parquet}

升级后原 v2.1 目录改名为 `<root>_v21` 保留；下次启动采集时
`restore_v21_if_needed` 会自动把它换回来继续续写，所以"采完即 v3.0、
再采自动回退"可以反复进行。
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:                    # pragma: no cover
    pa = None

V21 = "v2.1"
V30 = "v3.0"
CHUNK = 0
BATCH = "file-000"
BACKUP_SUFFIX = "_v21"
TMP_SUFFIX = ".v30tmp"
DISCARD_SUFFIX = ".v30discard"
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
VIDEO_NAMES = ["height", "width", "channels"]


# --------------------------------------------------------------- helpers

def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _version(root: pathlib.Path) -> str | None:
    info = root / "meta" / "info.json"
    if not info.exists():
        return None
    try:
        return json.loads(info.read_text(encoding="utf-8")).get(
            "codebase_version")
    except (OSError, ValueError):
        return None


def is_v30(root) -> bool:
    return _version(pathlib.Path(root).expanduser()) == V30


# --------------------------------------------------------------- stats

def _stats(arr: np.ndarray, axis, keepdims: bool) -> dict[str, np.ndarray]:
    """与 lerobot get_feature_stats 同构；count 统一记帧数。"""
    out = {
        "min": np.min(arr, axis=axis, keepdims=keepdims),
        "max": np.max(arr, axis=axis, keepdims=keepdims),
        "mean": np.mean(arr, axis=axis, keepdims=keepdims),
        "std": np.std(arr, axis=axis, keepdims=keepdims),
        "count": np.array([arr.shape[0]], dtype=np.int64),
    }
    for q in QUANTILES:
        out[f"q{int(q * 100):02d}"] = np.quantile(
            arr, q, axis=axis, keepdims=keepdims)
    return out


def _frames_of(mp4: pathlib.Path, want: int, tmpdir: pathlib.Path
               ) -> np.ndarray:
    """抽 want 帧并读成 (k, h, w, 3) 的 0~1 float32。"""
    from PIL import Image
    for old in tmpdir.glob("*.png"):
        old.unlink()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,width,height",
         "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True)
    total = 1
    fields = probe.stdout.strip().split(",")
    if fields and fields[0].isdigit():
        total = int(fields[0])
    step = max(1, total // max(1, want))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(mp4),
         "-vf", f"select='not(mod(n\\,{step}))'", "-vsync", "0",
         "-frames:v", str(want), str(tmpdir / "%03d.png")],
        check=True)
    arrs = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            for p in sorted(tmpdir.glob("*.png"))]
    for p in tmpdir.glob("*.png"):
        p.unlink()
    if not arrs:
        raise RuntimeError(f"抽帧失败: {mp4}")
    return np.stack(arrs)


def _video_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    """抽样帧 -> 每通道 (3,1,1)；与 lerobot 的 keepdims=True 对齐。"""
    st = _stats(arr, axis=(0, 1, 2), keepdims=True)
    for k in list(st):
        if k != "count":
            st[k] = np.asarray(st[k]).reshape(3, 1, 1)
    return st


def _stats_list(st: dict, key: str, *, is_count=False) -> list:
    """统计值 -> python list（写表时再按 count/float 落成 arrow 类型）。"""
    v = np.asarray(st[key]).reshape(-1)
    return [int(x) for x in v] if is_count else [float(x) for x in v]


# --------------------------------------------------------------- videos

def _concat_videos(parts: list[pathlib.Path], out: pathlib.Path) -> None:
    """无损拼接：concat demuxer + -c copy，失败则退回重编码。"""
    if len(parts) == 1:
        shutil.copyfile(parts[0], out)
        return
    listfile = out.with_suffix(".concat.txt")
    listfile.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in parts) + "\n",
        encoding="utf-8")
    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", str(listfile), "-c", "copy", "-movflags", "+faststart",
             str(out)], capture_output=True, text=True)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return
        print(f"[v30] 无损拼接失败，改重编码: {r.stderr.strip()[:200]}",
              flush=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", str(listfile), "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(out)], check=True)
    finally:
        listfile.unlink(missing_ok=True)


# --------------------------------------------------------------- convert

def convert_v21_to_v30(root, *, frame_samples: int = 16) -> dict:
    """就地升级；返回 {"skipped"/"episodes"/"frames"/"cameras"/"backup"}。"""
    if pa is None:
        raise RuntimeError("需要 pyarrow：pip install pyarrow")
    root = pathlib.Path(root).expanduser()
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"{root}/meta/info.json 不存在")
    if _version(root) == V30:
        return {"skipped": True, "root": str(root)}

    info = json.loads((root / "meta" / "info.json").read_text(
        encoding="utf-8"))
    if info.get("codebase_version") != V21:
        raise ValueError(f"只支持 v2.1，收到 {info.get('codebase_version')}")
    features = info["features"]
    cams = [k for k, v in features.items() if v.get("dtype") == "video"]
    fps = float(info["fps"])
    episodes = sorted(_read_jsonl(root / "meta" / "episodes.jsonl"),
                      key=lambda e: int(e["episode_index"]))
    tasks = _read_jsonl(root / "meta" / "tasks.jsonl")
    if not episodes:
        return {"skipped": True, "root": str(root), "reason": "空数据集"}

    tmp = root.with_name(root.name + TMP_SUFFIX)
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "data" / "chunk-000").mkdir(parents=True)
    (tmp / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    for cam in cams:
        (tmp / "videos" / cam / "chunk-000").mkdir(parents=True)

    # ---- 1. data：逐集读入，丢掉 video struct 列，重排全局 index 后拼接
    tables: list[pa.Table] = []
    offset = 0
    ep_meta: list[dict] = []
    for ep in episodes:
        i, n = int(ep["episode_index"]), int(ep["length"])
        t = pq.read_table(root / "data" / "chunk-000" /
                          f"episode_{i:06d}.parquet")
        drop = [c for c in t.column_names if c in cams]
        if drop:
            t = t.drop(drop)
        idx_col = t.schema.get_field_index("index")
        t = t.set_column(idx_col, "index",
                         pa.array(np.arange(offset, offset + n, dtype=np.int64)))
        tables.append(t)
        ep_meta.append({"episode_index": i, "length": n,
                        "from": offset, "to": offset + n})
        offset += n
    pq.write_table(pa.concat_tables(tables),
                   tmp / "data" / "chunk-000" / f"{BATCH}.parquet")

    # ---- 2. videos：每路相机按集顺序拼接，记录每集的时间区间
    cursor = 0.0
    for m in ep_meta:
        m["v_from"] = cursor
        m["v_to"] = cursor + m["length"] / fps
        cursor = m["v_to"]
    for cam in cams:
        parts = [root / "videos" / "chunk-000" / cam /
                 f"episode_{m['episode_index']:06d}.mp4" for m in ep_meta]
        missing = [str(p) for p in parts if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{cam} 缺视频: {missing[:3]}")
        _concat_videos(parts, tmp / "videos" / cam / "chunk-000" /
                       f"{BATCH}.mp4")
        print(f"[v30] 拼接 {cam}: {len(parts)} 段", flush=True)

    # ---- 3. 统计：标量用全量帧，视频用抽样帧
    scalar_keys = [k for k, v in features.items()
                   if v.get("dtype") == "float32"]

    scalar_data = {}
    for k in scalar_keys:
        scalar_data[k] = np.asarray(
            pq.read_table(tmp / "data" / "chunk-000" / f"{BATCH}.parquet")
            .column(k).to_pylist(), dtype=np.float32)
    global_stats: dict[str, dict] = {
        k: _stats(v, axis=0, keepdims=False) for k, v in scalar_data.items()}

    # 每集视频统计（抽帧后按通道算）
    scratch = tmp / "_frames"
    scratch.mkdir(exist_ok=True)
    ep_video_stats: dict[int, dict[str, dict]] = {}
    all_frames: list[np.ndarray] = []
    for m in ep_meta:
        per_cam = {}
        for cam in cams:
            mp4 = root / "videos" / "chunk-000" / cam / \
                f"episode_{m['episode_index']:06d}.mp4"
            arr = _frames_of(mp4, frame_samples, scratch)
            per_cam[cam] = _video_stats(arr)
            if len(all_frames) < 8:          # 全局视频统计只抽少量帧
                all_frames.append(arr[:4])
        ep_video_stats[m["episode_index"]] = per_cam
        print(f"[v30] 统计 episode {m['episode_index']}", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    for cam in cams:
        merged = np.concatenate([b for b in all_frames], axis=0)
        global_stats[cam] = _video_stats(merged)

    # ---- 4. episodes parquet（每集一行，含统计与视频时间戳）
    cols: dict[str, pa.Array] = {
        "episode_index": pa.array([m["episode_index"] for m in ep_meta],
                                  type=pa.int64()),
        "data/chunk_index": pa.array([CHUNK] * len(ep_meta), type=pa.int64()),
        "data/file_index": pa.array([0] * len(ep_meta), type=pa.int64()),
        "dataset_from_index": pa.array([m["from"] for m in ep_meta],
                                       type=pa.int64()),
        "dataset_to_index": pa.array([m["to"] for m in ep_meta],
                                     type=pa.int64()),
    }
    for cam in cams:
        cols[f"videos/{cam}/chunk_index"] = pa.array([CHUNK] * len(ep_meta),
                                                     type=pa.int64())
        cols[f"videos/{cam}/file_index"] = pa.array([0] * len(ep_meta),
                                                    type=pa.int64())
        cols[f"videos/{cam}/from_timestamp"] = pa.array(
            [m["v_from"] for m in ep_meta], type=pa.float64())
        cols[f"videos/{cam}/to_timestamp"] = pa.array(
            [m["v_to"] for m in ep_meta], type=pa.float64())

    task_of = {m["episode_index"]: ep.get("tasks", [])
               for m, ep in zip(ep_meta, episodes)}
    cols["tasks"] = pa.array([task_of[m["episode_index"]] for m in ep_meta],
                             type=pa.list_(pa.string()))
    cols["length"] = pa.array([m["length"] for m in ep_meta], type=pa.int64())

    for k in scalar_keys:
        data = scalar_data[k]
        for m in ep_meta:
            seg = data[m["from"]:m["to"]]
            st = _stats(seg, axis=0, keepdims=False)
            for name in st:
                cols.setdefault(f"stats/{k}/{name}", []).append(
                    _stats_list(st, name, is_count=(name == "count")))
    for cam in cams:
        for m in ep_meta:
            st = ep_video_stats[m["episode_index"]][cam]
            for name in st:
                cols.setdefault(f"stats/{cam}/{name}", []).append(
                    _stats_list(st, name, is_count=(name == "count")))

    cols["meta/episodes/chunk_index"] = pa.array([CHUNK] * len(ep_meta),
                                                 type=pa.int64())
    cols["meta/episodes/file_index"] = pa.array([0] * len(ep_meta),
                                                type=pa.int64())
    arrays = {}
    for name, value in cols.items():
        if name.startswith("stats/"):
            typ = (pa.list_(pa.int64()) if name.endswith("/count")
                   else pa.list_(pa.float32()))
            arrays[name] = pa.array(value, type=typ)
        else:
            arrays[name] = value
    pq.write_table(pa.table(arrays),
                   tmp / "meta" / "episodes" / "chunk-000" / f"{BATCH}.parquet")

    # ---- 5. tasks.parquet
    pq.write_table(
        pa.table({
            "task_index": pa.array([int(t.get("task_index", i)) for i, t in
                                    enumerate(tasks)], type=pa.int64()),
            "task": pa.array([str(t["task"]) for t in tasks],
                             type=pa.string()),
        }),
        tmp / "meta" / "tasks.parquet")

    # ---- 6. info.json（v3.0 字段：去掉 total_chunks，data/video 路径改成
    #      chunk/file 模板；视频特征补 names，否则 policy 侧读 names[2] 崩）
    v30_features = json.loads(json.dumps(features))
    for cam in cams:
        v30_features[cam]["names"] = VIDEO_NAMES
    v30_info = {
        "codebase_version": V30,
        "robot_type": info.get("robot_type", "unknown"),
        "total_episodes": len(ep_meta),
        "total_frames": offset,
        "total_tasks": len(tasks),
        "total_videos": len(ep_meta) * len(cams),
        "chunks_size": int(info.get("chunks_size", 1000)),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": ("videos/{video_key}/chunk-{chunk_index:03d}/"
                       "file-{file_index:03d}.mp4"),
        "fps": fps,
        "splits": {"train": f"0:{len(ep_meta)}"},
        "features": v30_features,
    }
    (tmp / "meta" / "info.json").write_text(
        json.dumps(v30_info, indent=2), encoding="utf-8")

    # ---- 7. stats.json（全局聚合）
    def _plain(st: dict) -> dict:
        return {k: np.asarray(v).tolist() for k, v in st.items()}

    (tmp / "meta" / "stats.json").write_text(
        json.dumps({k: _plain(v) for k, v in global_stats.items()}, indent=2),
        encoding="utf-8")

    # ---- 8. 原子换位：v2.1 留作备份，v3.0 占原名
    backup = root.with_name(root.name + BACKUP_SUFFIX)
    if backup.exists():
        shutil.rmtree(backup)
    root.rename(backup)
    tmp.rename(root)
    print(f"[v30] 升级完成：{len(ep_meta)} 集 / {offset} 帧 / "
          f"{len(cams)} 路相机；原 v2.1 备份在 {backup}", flush=True)
    return {"skipped": False, "root": str(root), "backup": str(backup),
            "episodes": len(ep_meta), "frames": offset, "cameras": cams}


def restore_v21_if_needed(root) -> bool:
    """上次退出时升级成了 v3.0：换回 v2.1 备份，让采集能继续续写。"""
    root = pathlib.Path(root).expanduser()
    backup = root.with_name(root.name + BACKUP_SUFFIX)
    if not backup.is_dir() or _version(root) != V30:
        return False
    discard = root.with_name(root.name + DISCARD_SUFFIX)
    if discard.exists():
        shutil.rmtree(discard)
    root.rename(discard)
    backup.rename(root)
    shutil.rmtree(discard, ignore_errors=True)
    print(f"[v30] 发现上次升级的 v3.0，已回退到 v2.1 备份继续采集",
          flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("用法: python -m simovia.lerobot_v30 <dataset_root>")
        return 2
    for arg in argv:
        print(convert_v21_to_v30(arg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
