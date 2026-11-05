"""Desktop/CLI entry point. Run: python -m demo.fruit_sorting"""
from __future__ import annotations
import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
if (HERE/'.deps').exists():sys.path.insert(0,str(HERE/'.deps'))
if sys.platform.startswith('linux'):os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,traceback,time
try:
    import cv2
except ImportError:
    raise SystemExit('需要 OpenCV：python -m pip install -r demo/fruit_sorting/requirements.txt')
import numpy as np
from .core import Demo,CONFIG,COLORS,detect,annotate
def dashboard(d):
    scene=d.overview()
    if d.calibration is not None and d.sim.spec.name=='fruit_sorting':
        d.perceive()
    rgb=d.rgb
    canvas=np.full((720,1280,3),(13,20,32),np.uint8)
    cv2.putText(canvas,'SIMOVIA  /  RGB FRUIT SORTING',(24,40),cv2.FONT_HERSHEY_SIMPLEX,.9,(231,239,250),2,cv2.LINE_AA)
    cv2.putText(canvas,d.stage+'   '+d.detail[:82],(24,72),cv2.FONT_HERSHEY_SIMPLEX,.51,(105,205,240),1,cv2.LINE_AA)
    canvas[98:658,:840]=cv2.resize(scene,(840,560))
    if rgb is not None:canvas[105:435,840:1280]=cv2.resize(rgb,(440,330))
    cv2.putText(canvas,'WRIST RGB / HSV DETECTIONS',(855,470),cv2.FONT_HERSHEY_SIMPLEX,.5,(225,233,244),1,cv2.LINE_AA)
    for i,k in enumerate(('banana','apple','grape')):
        n=sum(p['kind']==k for p in d.results)
        cv2.putText(canvas,f'{k.upper():7s}  {n}/2 placed',(864,510+i*38),cv2.FONT_HERSHEY_SIMPLEX,.65,COLORS[k],2,cv2.LINE_AA)
    if d.calibration:
        e=d.calibration['ground_truth_validation_only']
        cv2.putText(canvas,f"Hand-eye: {e['translation_m']*1000:.2f} mm / {e['rotation_deg']:.3f} deg",(24,692),cv2.FONT_HERSHEY_SIMPLEX,.55,(199,214,231),1,cv2.LINE_AA)
    return canvas
def headless(args):
    d=Demo(args.output,seed=args.seed);video=None
    try:
        if args.video:
            video=cv2.VideoWriter(str(Path(args.output)/'demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1280,720))
            if not video.isOpened():raise RuntimeError('Video writer unavailable')
        for i,_ in enumerate(d.run()):
            if video and i%10==0:
                video.write(cv2.cvtColor(dashboard(d),cv2.COLOR_RGB2BGR))
        if video:
            frame=cv2.cvtColor(dashboard(d),cv2.COLOR_RGB2BGR)
            for _ in range(20):video.write(frame)
        d.save_rgb('dashboard_final.png',dashboard(d))
    finally:
        if video:video.release()
        d.close()
def gui(args):
    from PySide6.QtCore import Qt,QTimer
    from PySide6.QtGui import QImage,QPixmap
    from PySide6.QtWidgets import QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QComboBox,QMessageBox,QSizePolicy
    app=QApplication([])
    class Window(QMainWindow):
        def __init__(self):
            super().__init__();self.setWindowTitle('Simovia · 机械臂视觉分拣 / 香蕉 · 苹果 · 葡萄')
            self.resize(1280,860);self.demo=Demo(args.output);self.iterator=None;self.count=0;self.running=False
            central=QWidget();self.setCentralWidget(central);lay=QVBoxLayout(central)
            self.image=QLabel();self.image.setAlignment(Qt.AlignmentFlag.AlignCenter);self.image.setMinimumSize(640,360);self.image.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Expanding);lay.addWidget(self.image,1)
            bar=QHBoxLayout();lay.addLayout(bar)
            self.start=QPushButton('开始：手眼标定 → 识别 → 分拣');self.pause=QPushButton('暂停');self.reset=QPushButton('重置');self.speed=QComboBox();self.speed.addItems(['1× 实时','2×','4×','8×']);self.speed.setCurrentIndex(1)
            for w in (self.start,self.pause,self.reset,self.speed):bar.addWidget(w)
            self.info=QLabel('Panda + 2FG7 · 腕部 RGB 相机 · 每种 2 个 · 红苹果 / 黄香蕉 / 紫葡萄');self.info.setWordWrap(True);lay.addWidget(self.info)
            self.statusBar().showMessage('请点击开始。先标定，再进入分拣场景；全部动作使用物理仿真。')
            self.start.clicked.connect(self.begin);self.pause.clicked.connect(self.toggle);self.reset.clicked.connect(self.restart)
            self.timer=QTimer(self);self.timer.timeout.connect(self.advance);self.timer.start(20);self.refresh()
        def begin(self):
            if self.iterator is None:self.iterator=self.demo.run()
            self.running=True;self.start.setEnabled(False);self.pause.setText('暂停')
        def toggle(self):
            if self.iterator is None:return
            self.running=not self.running;self.pause.setText('暂停' if self.running else '继续')
        def restart(self):
            self.running=False;self.iterator=None;self.demo.close();self.demo=Demo(args.output);self.start.setEnabled(True);self.pause.setText('暂停');self.refresh()
        def advance(self):
            if not self.running:return
            try:
                for _ in range(2**self.speed.currentIndex()):
                    next(self.iterator);self.count+=1
                if self.count%5 < 2**self.speed.currentIndex():self.refresh()
            except StopIteration:
                self.running=False;self.iterator=None;self.refresh();self.statusBar().showMessage('完成：已独立核对各水果的最终落框位置。点击重置可重新演示。')
            except Exception as e:
                self.running=False;self.statusBar().showMessage(str(e));QMessageBox.critical(self,'演示停止',str(e));traceback.print_exc()
        def refresh(self):
            rgb=dashboard(self.demo);h,w,_=rgb.shape
            q=QImage(rgb.data,w,h,3*w,QImage.Format.Format_RGB888).copy()
            self.image.setPixmap(QPixmap.fromImage(q).scaled(self.image.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
            labels={'CALIBRATION':'采集手眼标定图','CALIBRATED':'标定完成','OBSERVE':'进入生产场景','DETECT':'HSV识别水果','PICK':'抓取','PLACE':'放入对应分拣框','VERIFY':'核对结果','COMPLETE':'分拣完成','FAILED':'分拣未全部成功','Ready':'等待开始'}
            self.info.setText(labels.get(self.demo.stage,self.demo.stage)+'  |  '+self.demo.detail+'  |  香蕉 BANANA · 苹果 APPLE · 葡萄 GRAPE')
        def resizeEvent(self,event):
            super().resizeEvent(event)
            if hasattr(self,'demo'):self.refresh()
        def closeEvent(self,event):
            self.timer.stop();self.demo.close();event.accept()
    app.setStyle('Fusion')
    app.setStyleSheet('QWidget{background:#111b2b;color:#e5edf7;font-size:14px;}QPushButton,QComboBox{padding:10px;border:1px solid #45617e;border-radius:6px;background:#20334b;}QPushButton:disabled{color:#708397;}')
    w=Window();w.show()
    if args.autostart:QTimer.singleShot(300,w.begin)
    sys.exit(app.exec())
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=0,help='Nonzero adds +/-3mm initial fruit-position jitter')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--video',action='store_true',help='Headless: record a 2x simulation-time overview + annotated RGB movie')
    parser.add_argument('--autostart',action='store_true')
    parser.add_argument('--output',default=str(HERE/'runs'/'latest'))
    args=parser.parse_args()
    (headless if args.headless else gui)(args)
if __name__=='__main__':main()
