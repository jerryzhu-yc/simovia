"""Image-only hand-eye estimation and checked reuse of persisted calibration."""
from dataclasses import asdict
from pathlib import Path
import hashlib, json
import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
TOOL_FRAME = {'type': 'site', 'name': 'gripper/gripper_tcp'}

def transform(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(t).reshape(3)
    return T

def inverse(T):
    return transform(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])

def intrinsics(fovy, width=WIDTH, height=HEIGHT):
    f = height / (2 * np.tan(np.radians(fovy) / 2))
    return np.array([[f, 0, (width-1)/2], [0, f, (height-1)/2], [0, 0, 1.]])

def rig_identity(spec, robot_name, camera_name, tool_frame, width=WIDTH, height=HEIGHT):
    """Identify the assembly, not world placement or calibration-board position."""
    robot = next(r for r in spec.robots if r.name == robot_name)
    camera = next(s for s in robot.sensors if s.name == camera_name and s.type == 'camera')
    def asset(value):
        if value is None: return None
        p = (spec.base_dir / value).resolve()
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else str(value)
    ees = []
    for ee in robot.end_effectors:
        item = asdict(ee); item['model'] = asset(ee.model); ees.append(item)
    identity = dict(model=asset(robot.model), inline_spec=robot.spec,
                    end_effectors=ees, camera=asdict(camera), tool_frame=tool_frame,
                    resolution=[width, height], strip_hands=robot.strip_hands)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return dict(sha256=digest, assembly=identity)

def detect_target(rgb, corners_by_id, K):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 3
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None: raise ValueError('没有检测到标定板，请调整相机朝向和采样姿态')
    selected = [(c, int(i)) for c, i in zip(corners, ids.ravel()) if str(int(i)) in corners_by_id]
    if len(selected) < 6: raise ValueError('可用标记不足 6 个，请检查板面遮挡和视野')
    obj = np.concatenate([np.array(corners_by_id[str(i)], np.float32) for c, i in selected])
    img = np.concatenate([c.reshape(4, 2) for c, i in selected])
    ok, rv, tv = cv2.solvePnP(obj, img, K, np.zeros(5))
    if not ok or tv[2, 0] <= 0: raise ValueError('标定板 PnP 求解失败')
    proj = cv2.projectPoints(obj, rv, tv, K, np.zeros(5))[0].reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum((proj-img)**2, axis=1))))
    return transform(cv2.Rodrigues(rv)[0], tv), error, cv2.aruco.drawDetectedMarkers(rgb.copy(), corners, ids), len(selected)

def solve(samples, K, identity, profile):
    if len(samples) < 12: raise ValueError('至少需要 12 个样本，其中最后 2 个用于留出验证')
    G = [np.array(s['gripper_to_world']) for s in samples]
    C = [np.array(s['target_to_camera']) for s in samples]
    train = len(samples)-2
    rotations = np.array([cv2.Rodrigues(g[:3,:3] @ G[0][:3,:3].T)[0].ravel() for g in G[1:train]])
    if np.linalg.svd(rotations, compute_uv=False)[1] < .05:
        raise ValueError('姿态退化：需围绕至少两个不同轴旋转，不能仅平移或单轴转动')
    R, t = cv2.calibrateHandEye([g[:3,:3] for g in G[:train]], [g[:3,3] for g in G[:train]],
                              [c[:3,:3] for c in C[:train]], [c[:3,3] for c in C[:train]], method=cv2.CALIB_HAND_EYE_PARK)
    X = transform(R, t)
    if not np.isfinite(X).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or np.linalg.det(R) < .999:
        raise ValueError('手眼解无效')
    boards = [g @ X @ c for g, c in zip(G, C)]
    center = np.mean([b[:3,3] for b in boards[:train]], axis=0)
    residuals = [float(np.linalg.norm(b[:3,3]-center)) for b in boards]
    rotation_residuals = [float(np.linalg.norm(cv2.Rodrigues(b[:3,:3] @ boards[0][:3,:3].T)[0])*180/np.pi) for b in boards]
    repro = float(np.mean([s['reprojection_px'] for s in samples]))
    if max(residuals[-2:]) > .008 or max(rotation_residuals[-2:]) > 2 or repro > 1:
        raise ValueError('标定未通过留出验证：位置须 ≤8mm，旋转 ≤2°，平均重投影 ≤1px')
    return dict(format_version=1, mode='eye_in_hand', profile=profile,
                method='OpenCV PARK AX=XB; rendered ArUco + solvePnP',
                camera_axes='OpenCV: x right, y down, z forward',
                resolution=identity['assembly']['resolution'], rig_identity=identity,
                intrinsics=K.tolist(), distortion=[0]*5, gripper_T_camera=X.tolist(),
                train_samples=train, held_out_samples=2, reprojection_px_mean=repro,
                board_position_residuals_m=residuals, board_rotation_residuals_deg=rotation_residuals,
                quality={'validated': True, 'held_out_max_m': max(residuals[-2:])}, samples=samples)

def save_result(path, result):
    """Publish only validated results; a failed run cannot replace a working file."""
    if not result.get('quality', {}).get('validated'): raise ValueError('拒绝保存未通过验证的标定')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    temp.replace(path)

def load_result(path, spec, robot_name='arm', camera_name='wrist_rgb', tool_frame=None):
    path = Path(path)
    if not path.is_file(): raise ValueError(f'缺少标定文件：{path}。请先运行独立手眼标定 demo')
    result = json.loads(path.read_text(encoding='utf8'))
    if result.get('format_version') != 1 or result.get('mode') != 'eye_in_hand' or not result.get('quality', {}).get('validated'):
        raise ValueError('标定文件版本或质量状态无效，请重新标定')
    expected = rig_identity(spec, robot_name, camera_name, tool_frame or TOOL_FRAME)
    if result.get('rig_identity', {}).get('sha256') != expected['sha256']:
        raise ValueError('标定与当前机械臂、夹爪或相机配置不匹配，请为当前装配重新标定')
    X = np.asarray(result['gripper_T_camera'], dtype=float)
    K = np.asarray(result['intrinsics'], dtype=float)
    camera = next(s for r in spec.robots if r.name == robot_name for s in r.sensors if s.name == camera_name)
    if X.shape != (4,4) or not np.isfinite(X).all() or not np.allclose(X[3], [0,0,0,1]) or not np.allclose(X[:3,:3].T @ X[:3,:3], np.eye(3), atol=1e-5) or np.linalg.det(X[:3,:3]) < .999:
        raise ValueError('标定矩阵不是有效刚体变换')
    if K.shape != (3,3) or not np.allclose(K, intrinsics(camera.fovy or 45)) or result.get('resolution') != [WIDTH, HEIGHT]:
        raise ValueError('相机内参或分辨率与当前渲染配置不一致')
    if not np.allclose(np.asarray(result.get('distortion'), dtype=float), np.zeros(5)):
        raise ValueError('当前仿真 RGB 相机要求零畸变标定')
    return result
