"""Desktop mission console, launched from Simovia's File menu."""
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import Qt,QTimer
from PySide6.QtGui import QImage,QPixmap
from PySide6.QtWidgets import (QMainWindow,QWidget,QLabel,QPushButton,QComboBox,
    QVBoxLayout,QHBoxLayout,QFrame,QProgressBar,QTextEdit,QSizePolicy)
from .core import Demo
from .view import View
from .scene import HERE

STAGES={
    'READY':('准备就绪',0),'STABILIZE':('站立与装载准备',1),'PICK':('选择维修模块',2),
    'LOAD':('装载交接',3),'DELIVER':('跨区配送',4),'WAIT_STATION':('等待工位',5),
    'UNLOAD':('到站卸载',6),'INSTALL':('模块安装',7),'RESET_DEVICE':('复位验证',8),
    'COMPLETE':('设备恢复',9),'FAILED':('任务停止',0),'CANCELLED':('已取消',0),
}
STYLES="""
QWidget{background:#0c1420;color:#e5edf7;font-family:'Noto Sans CJK SC','Microsoft YaHei',sans-serif;font-size:13px;}
QLabel#eyebrow{color:#51d6c0;font-size:12px;font-weight:700;}
QLabel#heading{font-size:24px;font-weight:700;}
QLabel#muted{color:#93a5b9;font-size:12px;}
QFrame#card{background:#142234;border:1px solid #273a50;border-radius:8px;}
QFrame#card QLabel{background:transparent;}
QLabel#cardtitle{font-size:15px;font-weight:700;}
QPushButton,QComboBox{background:#1c3047;border:1px solid #3c536c;border-radius:6px;padding:9px 14px;}
QPushButton:hover{background:#29465e;}
QPushButton:disabled{color:#63768c;background:#17212f;}
QPushButton#primary:disabled{background:#203448;border:1px solid #34475c;color:#8294a8;}
QPushButton#primary{background:#167e73;border:1px solid #29b7a2;color:white;font-weight:700;}
QProgressBar{border:0;background:#25364b;border-radius:4px;min-height:7px;max-height:7px;}
QProgressBar::chunk{background:#37c8ad;border-radius:4px;}
QTextEdit{background:#101c2c;border:1px solid #273a50;border-radius:6px;padding:6px;font-size:12px;}
"""

class SceneLabel(QLabel):
    def __init__(self,window):
        super().__init__();self.window=window;self.last=None
        self.setMinimumSize(640,420)
        self.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
    def mousePressEvent(self,event):self.last=event.position()
    def mouseReleaseEvent(self,event):self.last=None
    def mouseMoveEvent(self,event):
        if self.last is None:return
        delta=event.position()-self.last;self.last=event.position()
        self.window.view.mode='自由视角'
        self.window.view.cam.azimuth-=delta.x()*.3
        self.window.view.cam.elevation=max(-88,min(-5,self.window.view.cam.elevation-delta.y()*.2))
        self.window.refresh()
    def wheelEvent(self,event):
        self.window.view.cam.distance=max(.7,min(8,self.window.view.cam.distance*(.9 if event.angleDelta().y()>0 else 1.1)))
        self.window.refresh()

class Window(QMainWindow):
    def __init__(self,output=None,parent=None,station_delay=2.):
        super().__init__(parent,Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle('Simovia · Go2 × G1 × UR5e 联合维修')
        self.resize(1420,890);self.setMinimumSize(1100,720);self.setStyleSheet(STYLES)
        self.output=Path(output) if output else HERE/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S')
        self.station_delay=station_delay
        self.episode=0;self.running=False;self.iterator=None;self.count=0;self.event_count=0
        self.demo=Demo(self.output/'episode_00',station_delay=self.station_delay);self.view=View(self.demo)
        central=QWidget();self.setCentralWidget(central);layout=QVBoxLayout(central);layout.setContentsMargins(22,18,22,16);layout.setSpacing(12)
        eyebrow=QLabel('SIMOVIA  /  HETEROGENEOUS ROBOT TEAM');eyebrow.setObjectName('eyebrow');layout.addWidget(eyebrow)
        title=QLabel('无人厂区 · 三机联合维修');title.setObjectName('heading');layout.addWidget(title)
        subtitle=QLabel('UR5e 备料装载  →  Go2 跨区配送  →  G1 取件安装与设备复位');subtitle.setObjectName('muted');layout.addWidget(subtitle)
        body=QHBoxLayout();body.setSpacing(18);layout.addLayout(body,1)
        self.image=SceneLabel(self);body.addWidget(self.image,1)
        side=QVBoxLayout();side.setSpacing(10);body.addLayout(side)
        self.phase=QLabel('准备就绪');self.phase.setObjectName('cardtitle');side.addWidget(self.phase)
        self.progress=QProgressBar();self.progress.setRange(0,9);self.progress.setTextVisible(False);side.addWidget(self.progress)
        self.cards={}
        for key,title,note in [('loader','01  UR5e · 备料员','六轴机械臂 / 平行夹爪'),('courier','02  Go2 · 配送员','四足机器人 / 限位运输架'),('technician','03  G1 · 维修员','人形机器人 / 站立作业')]:
            frame=QFrame();frame.setObjectName('card');frame.setFixedWidth(300)
            box=QVBoxLayout(frame);box.setContentsMargins(14,12,14,12)
            name=QLabel(title);name.setObjectName('cardtitle');box.addWidget(name)
            desc=QLabel(note);desc.setObjectName('muted');box.addWidget(desc)
            status=QLabel('等待任务');status.setWordWrap(True);box.addWidget(status);self.cards[key]=status
            side.addWidget(frame)
        self.metrics=QLabel();self.metrics.setWordWrap(True);side.addWidget(self.metrics)
        self.log=QTextEdit();self.log.setReadOnly(True);self.log.setFixedWidth(300);side.addWidget(self.log,1)
        bar=QHBoxLayout();layout.addLayout(bar)
        self.start=QPushButton('开始联合任务');self.start.setObjectName('primary')
        self.pause=QPushButton('暂停');self.pause.setEnabled(False)
        self.step_button=QPushButton('单步');self.step_button.setEnabled(False)
        self.reset=QPushButton('重置场景')
        self.camera=QComboBox();self.camera.addItems(['总览','装载工位','跟随 Go2','维修工位','俯视'])
        self.speed=QComboBox();self.speed.addItems(['1×','2×','4×'])
        for w in (self.start,self.pause,self.step_button,self.reset):bar.addWidget(w)
        bar.addStretch();bar.addWidget(self.camera);bar.addWidget(self.speed)
        self.footer=QLabel('拖动画面旋转视角，滚轮缩放。蓝色模块到位且按钮实际触发后，任务才会完成。')
        self.footer.setObjectName('muted');layout.addWidget(self.footer)
        self.start.clicked.connect(self.begin);self.pause.clicked.connect(self.toggle)
        self.step_button.clicked.connect(lambda:self.advance(single=True));self.reset.clicked.connect(self.restart)
        self.camera.currentTextChanged.connect(self.set_camera)
        self.timer=QTimer(self);self.timer.timeout.connect(self.advance);self.timer.start(16)
        self.refresh()

    def begin(self):
        if self.demo.done:return
        if self.iterator is None:self.iterator=self.demo.run()
        self.running=True;self.start.setEnabled(False);self.pause.setEnabled(True);self.pause.setText('暂停');self.step_button.setEnabled(False)

    def toggle(self):
        if self.iterator is None or self.demo.done:return
        self.running=not self.running
        self.pause.setText('暂停' if self.running else '继续');self.step_button.setEnabled(not self.running)
        self.refresh()

    def set_camera(self,name):
        self.view.preset(name);self.refresh()

    def advance(self,single=False):
        if self.iterator is None or (not self.running and not single):return
        try:
            for _ in range(1 if single else 2**self.speed.currentIndex()):
                next(self.iterator);self.count+=1
            if single or self.count%3<2**self.speed.currentIndex():self.refresh()
        except StopIteration:
            self.running=False;self.iterator=None;self.pause.setEnabled(False);self.step_button.setEnabled(False);self.refresh()
        except Exception:
            self.running=False;self.iterator=None;self.pause.setEnabled(False);self.step_button.setEnabled(False);self.refresh()

    def restart(self):
        self.running=False
        if self.iterator is not None:
            self.demo.transition('CANCELLED','用户重置场景');self.iterator.close();self.iterator=None
        self.view.close();self.demo.close();self.episode+=1
        self.demo=Demo(self.output/f'episode_{self.episode:02d}',station_delay=self.station_delay);self.view=View(self.demo)
        self.view.preset(self.camera.currentText());self.event_count=0;self.log.clear();self.count=0
        self.start.setEnabled(True);self.pause.setEnabled(False);self.pause.setText('暂停');self.step_button.setEnabled(False)
        self.footer.setText('拖动画面旋转视角，滚轮缩放。蓝色模块到位且按钮实际触发后，任务才会完成。');self.refresh()

    def refresh(self):
        rgb=self.view.render();h,w,_=rgb.shape
        q=QImage(rgb.data,w,h,3*w,QImage.Format.Format_RGB888).copy()
        self.image.setPixmap(QPixmap.fromImage(q).scaled(self.image.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        stage=self.demo.stage;label,progress=STAGES[stage]
        suffix=' · 已暂停' if self.iterator is not None and not self.running else ''
        self.phase.setText(label+suffix);self.progress.setValue(progress)
        statuses={
            'loader':'待机 · 等待装载任务','courier':'站立保持 · 等待交接','technician':'站立平衡 · 等待模块',
        }
        if stage in ('PICK','LOAD'):statuses['loader']='取料中' if stage=='PICK' else '装载中 · Go2 保持静止'
        if stage in ('DELIVER','WAIT_STATION','UNLOAD','INSTALL','RESET_DEVICE','COMPLETE'):statuses['loader']='已退出交接区 · 待机'
        if stage=='DELIVER':statuses['courier']='配送中 · 航向反馈控制'
        elif stage=='WAIT_STATION':statuses['courier']='到站等待 · 工位尚未就绪'
        elif stage in ('UNLOAD','INSTALL','RESET_DEVICE'):statuses['courier']='停靠保持 · 等待卸载确认'
        if stage in ('UNLOAD','INSTALL','RESET_DEVICE'):statuses['technician']={'UNLOAD':'取件中 · Go2 已停稳','INSTALL':'模块安装中','RESET_DEVICE':'接触复位按钮并核验'}[stage]
        if stage=='COMPLETE':statuses={k:'任务完成 · 保持待机' for k in statuses}
        if stage=='FAILED':statuses={k:'已停止 · 查看任务日志' for k in statuses}
        for k,v in statuses.items():self.cards[k].setText(v)
        self.metrics.setText(f'仿真时间  {self.demo.data.time:05.1f} s\n模块位置  {self.demo.owner}\n设备状态  '+('运行正常' if self.demo.success else '等待维修')+f'\n机器人间接触峰值  {self.demo.collision_peak:.1f} N')
        for event in self.demo.events[self.event_count:]:
            self.log.append(f"{event['time']:05.1f}s  {event['detail']}")
        self.event_count=len(self.demo.events)
        if stage=='FAILED':self.footer.setText('任务中止：'+self.demo.detail+'。可重置后重新运行。')

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'view') and hasattr(self,'phase'):self.refresh()

    def closeEvent(self,event):
        self.timer.stop();self.running=False
        if self.iterator is not None:
            self.demo.transition('CANCELLED','用户关闭演示');self.iterator.close();self.iterator=None
        self.view.close();self.demo.close();event.accept()

def create_window(output=None,parent=None,station_delay=2.):
    return Window(output=output,parent=parent,station_delay=station_delay)
