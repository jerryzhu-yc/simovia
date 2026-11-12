"""MuJoCo overview and follow cameras shared by the desktop and recorder."""
import numpy as np
import mujoco

class View:
    def __init__(self,demo,width=1100,height=720):
        self.demo=demo;self.width=width;self.height=height
        m=demo.model
        m.vis.global_.offwidth=max(width,m.vis.global_.offwidth)
        m.vis.global_.offheight=max(height,m.vis.global_.offheight)
        self.renderer=mujoco.Renderer(m,height,width)
        self.cam=mujoco.MjvCamera()
        self.mode='总览'
        self.preset('总览')

    def preset(self,mode):
        self.mode=mode
        if mode=='俯视':
            target=[.85,.05,.1];distance=3.7;azimuth=90;elevation=-86
        elif mode=='装载工位':
            target=[-.15,.1,.5];distance=1.75;azimuth=125;elevation=-28
        elif mode=='维修工位':
            target=[1.85,.25,.65];distance=2.3;azimuth=30;elevation=-25
        elif mode=='跟随 Go2':
            target=self.demo.data.xpos[self.demo.dog.base]+[0,0,.2];distance=1.7;azimuth=120;elevation=-22
        else:
            target=[.85,.12,.6];distance=4.6;azimuth=118;elevation=-32
        self.cam.lookat[:]=target;self.cam.distance=distance
        self.cam.azimuth=azimuth;self.cam.elevation=elevation

    def render(self):
        if self.mode=='跟随 Go2':
            self.cam.lookat[:]=self.demo.data.xpos[self.demo.dog.base]+[0,0,.2]
        opt=mujoco.MjvOption()
        opt.sitegroup[:]=0
        self.renderer.update_scene(self.demo.data,camera=self.cam,scene_option=opt)
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()
