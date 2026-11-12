"""Run the relay, or validate it without a desktop."""
import argparse,os,sys,json,math
from pathlib import Path
from datetime import datetime

def main():
    parser=argparse.ArgumentParser(description='Simovia Go2 / G1 / UR5e maintenance relay')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--video',action='store_true',help='Record a headless run (requires OpenCV)')
    parser.add_argument('--autostart',action='store_true')
    parser.add_argument('--station-delay',type=float,default=2.)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not math.isfinite(args.station_delay) or args.station_delay<0 or args.station_delay>30:parser.error('--station-delay must be between 0 and 30 seconds')
    if args.video and not args.headless:parser.error('--video requires --headless')
    if sys.platform.startswith('linux'):os.environ.setdefault('MUJOCO_GL','egl')
    from .scene import HERE
    output=args.output or HERE/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.headless:
        from .core import Demo
        demo=Demo(output,station_delay=args.station_delay)
        view=writer=None
        try:
            if args.video:
                import cv2
                from .view import View
                output.mkdir(parents=True,exist_ok=True)
                view=View(demo,1100,720)
                writer=cv2.VideoWriter(str(output/'relay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),15,(1100,720))
                if not writer.isOpened():raise RuntimeError('无法创建演示录像')
            camera_stage=None
            for i,_ in enumerate(demo.run()):
                if view is not None and demo.stage!=camera_stage:
                    camera_stage=demo.stage
                    mode=('装载工位' if demo.stage in ('PICK','LOAD') else '跟随 Go2' if demo.stage=='DELIVER' else '维修工位' if demo.stage in ('WAIT_STATION','UNLOAD','INSTALL','RESET_DEVICE') else '总览')
                    view.preset(mode)
                if writer is not None and i%4==0:
                    rgb=view.render()
                    cv2.putText(rgb,f'SIMOVIA  /  {demo.stage}  /  {demo.data.time:.1f}s',(24,36),cv2.FONT_HERSHEY_SIMPLEX,.75,(12,36,52),2,cv2.LINE_AA)
                    writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            if view is not None:
                view.preset('总览')
                rgb=view.render()
                cv2.imwrite(str(output/'final.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                for _ in range(20):writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            print(json.dumps(dict(success=demo.success,result=str(output/'result.json')),ensure_ascii=False))
            return 0 if demo.success else 1
        except Exception as error:
            print(str(error),file=sys.stderr);return 1
        finally:
            if writer is not None:writer.release()
            if view is not None:view.close()
            demo.close()
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from .gui import create_window
    app=QApplication.instance() or QApplication(sys.argv)
    window=create_window(output,station_delay=args.station_delay);window.show()
    if args.autostart:QTimer.singleShot(200,window.begin)
    return app.exec()

if __name__=='__main__':
    raise SystemExit(main())
