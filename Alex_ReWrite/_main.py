import numpy as np
import pygame
import pyglet
from Drawer import Drawer
from Globals import displayHeight, displayWidth
from PygameAdditionalMethods import *
from ShapeObjects import Line
from pyglet.window import key
import math, random, time, os

# SINUS ist Gegenkathete!!

drawer = Drawer()
vec2 = pygame.math.Vector2

class Game:
    def __init__(self):
        trackImg = pyglet.image.load('images/track.png')
        self.winWidth = int(trackImg.width * 0.6)
        self.winHeight = int(trackImg.height * 0.6)
        self.trackSprite = pyglet.sprite.Sprite(trackImg, x=0, y=0)
        self.trackSprite.update(scale=0.6)
        self.car = Car()

    def render(self):
        #glPushMatrix()
        self.trackSprite.draw()
        self.car.carSprite.draw()
        #self.car.setWalls()
        self.renderWalls()
        self.renderRewardGates()
        self.renderVisionVectors()
        #glPopMatrix()

    def renderVisionVectors(self):
        self.car.visionVector = []
        self.car.setVisionVector()
        for visionGate in self.car.visionVector:
            visionGate.line.draw()

    def renderWalls(self):
        for wall in self.car.walls:
            wall.line.draw()
        
    def renderRewardGates(self):
        for rewardGate in self.car.rewardGates:
            rewardGate.line.draw()

class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.line = Line(x1,y1, x2, y2)
        self.line.setColor([0, 0, 0])

class RewardGate:
    def __init__(self, x1, y1, x2, y2):
        self.line = Line(x1, y1, x2, y2)
        self.line.setColor([0, 255, 0])

class Car:
    def __init__(self):
        self.x = 174
        self.y = 141
        self.scalefactor = 0.6
        self.width = int(40*self.scalefactor)
        self.height = int(20*self.scalefactor)
        self.carPic = pyglet.image.load('images/car.png')
        self.carPic.anchor_x = self.carPic.width//2
        self.carPic.anchor_y = self.carPic.height//2
        self.carSprite = pyglet.sprite.Sprite(self.carPic, x=self.x, y=self.y)
        self.carSprite.opacity =80
        self.carSprite.update(rotation=0, scale_x=self.width / self.carSprite.width,
                                scale_y=self.height / self.carSprite.height)
        self.accInc = 2
        self.maxVelocity = 20

        #self.speedVec = [0,100/180*math.pi] # [amplitude, angle]
        self.speedVec = [0,90/180*math.pi] # [amplitude, angle]
        self.acceleration = [0,0] # [-1,1] means Deceleration and Left, [0,-1] means No Speed Delta and Right

        self.walls = []
        self.setWalls()

        self.rewardGates = []
        self.setRewardGates()
        
        self.visionVector = []
        self.setVisionVector()

        #self.speedVec = [0,100/180*math.pi] # [amplitude, angle]
        #self.acceleration = [0,0] # [-1,1] means Deceleration and Left, [0,-1] means No Speed Delta and Right
        
        self.reward = 0
        self.isDead = False

    def setVisionVector(self):
        
        self.visionVector = []
        angle_car = self.speedVec[1]
        front_x = self.x+math.cos(angle_car)*self.width/2
        front_y = self.y+math.sin(angle_car)*self.width/2

        x1 = front_x
        y1 = front_y #print(front_x, front_y) front_y is really Y-Coordinate, and vice versa for front_x
        x2,y2 = self.getDeltasforVision(self.speedVec[1]+0/180*math.pi,100)
        self.visionVector.append(Wall(x1, y1, x1+x2,y1+y2))

        x2 = front_x + 3*math.sin(math.pi- angle_car)
        y2 = front_y + 3*math.cos(math.pi- angle_car)
        deltaX,deltaY = self.getDeltasforVision(self.speedVec[1]-30/180*math.pi,100)
        self.visionVector.append(Wall(x2, y2, x2+deltaX,y2+deltaY))

        x3 = front_x - 3*math.sin(math.pi- angle_car)
        y3 = front_y - 3*math.cos(math.pi- angle_car)
        deltaX,deltaY = self.getDeltasforVision(self.speedVec[1]+30/180*math.pi,100)
        self.visionVector.append(Wall(x3, y3, x3+deltaX,y3+deltaY))
        # x3 = self.x + 4
        # y3 = self.y+self.width/2
        # x4,y4 = self.getDeltasforVision(self.speedVec[1]-30/180*math.pi,100)
        # self.visionVector.append(Wall(x3, y3, x3+x4,y3+y4))

    def setWalls(self):
        self.walls.append(Wall(226, 366, 104, 369))
        

    def getDeltasforVision(self,angle, length):
        # Considering the normal axis of the car, give me arbitrary deltaX and deltaY
        # for a vector with length length and  angle angle
        # Tuple returned can be used to define Vision Vector Coordinates
        x = math.cos(angle)*length
        y = math.sin(angle)*length
        return (x,y)

    def setRewardGates(self):
        self.rewardGates.append(RewardGate(107, 341, 210, 321))

    def polar_to_cartesian(self, polar_coords):
        amplitude, angle = polar_coords
        x = amplitude * math.cos(angle)
        y = amplitude * math.sin(angle)
        return (x, y)
    
    def newPosition(self):
        # CHANGE IN VELOCITY (Direction or Amplitude)
        if self.acceleration[1] != 0:
            self.speedVec[1] += self.acceleration[1]*math.pi/10
        if self.acceleration[0] != 0:
            self.speedVec[0] += self.acceleration[0]*2
            if abs(self.speedVec[0]) > self.maxVelocity:
                a = self.speedVec[0]
                self.speedVec[0] = a/abs(a)*self.maxVelocity #Cheeky bastard can't cheat by going backwards
        else:
            self.speedVec[0] *= 0.90

        # CHANGE IN POSITION        
        deltaX, deltaY = self.polar_to_cartesian(self.speedVec)     
        self.x += deltaX
        self.y += deltaY
        
        if self.x >= 1000 or self.x <= 0:
            self.x = 0
        if self.y >= 600 or self.y <= 0:
            self.y = 0

    def getCoordinatesFromVisionVector(self):
        angle = self.speedVec[1]
        x = self.x
        y = self.y
        length = 20
        x1 = math.cos(angle)*length+x
        y1 = math.sin(angle)*length+y
        return (x1, y1)
        
    def getState(self):
        pass

    def update(self):
        self.newPosition()
        self.checkWallHit()
        self.checkRewardGateHit()
        self.calcCollisionDistances()
        self.calcRewardGateDistance()
        self.carSprite.update(rotation=self.speedVec[1]/(2*math.pi)*-360, x=self.x, y=self.y)
        if self.isDead:
            return
        else:
            return self.getState()

    def checkWallHit(self):
        pass

    def checkRewardGateHit(self):
        pass
    
    def actionFromNumber(self, nr): 
        if nr == 9:
            return [1,-1]
        elif nr == 8:
            return [1,0]
        elif nr == 7:
            return [1,1]
        elif nr == 6:
            return [-1,-1]
        elif nr == 5:
            return [-1,0]
        elif nr == 4:
            return [-1,1]
        elif nr == 3:
            return [0,-1]
        elif nr == 2:
            return [0,0]
        elif nr == 1:
            return [0,1]

    def calcCollisionDistances(self):
        pass

    def calcRewardGateDistance(self):
        pass

    def reset(self):
        pass

    def observe(self):
        pass

class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game = Game()
        self.firstClick = True
        self.set_minimum_size(self.game.winWidth, self.game.winHeight)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.LEFT:
            self.game.car.acceleration[1] = 1
        if symbol == key.RIGHT:
            self.game.car.acceleration[1] = -1
        if symbol == key.UP:
            self.game.car.acceleration[0] = 1
        if symbol == key.DOWN:
            self.game.car.acceleration[0] = -1

    def on_key_release(self, symbol, modifiers):
        if symbol == key.LEFT:
            self.game.car.acceleration[1] = 0
        if symbol == key.RIGHT:
            self.game.car.acceleration[1] = 0
        if symbol == key.UP:
            self.game.car.acceleration[0] = 0
        if symbol == key.DOWN:
            self.game.car.acceleration[0] = 0
    def on_mouse_press(self, x, y, button, modifiers):
        if self.firstClick:
            self.clickPos = [x, y]
        else:
            print("self.walls.append(Wall({}, {}, {}, {}))".format(self.clickPos[0],
                                                                    displayHeight - self.clickPos[1],
                                                                    x, displayHeight - y))
        #
            # self.gates.append(RewardGate(self.clickPos[0], self.clickPos[1], x, y))
        
        self.firstClick = not self.firstClick
    def update(self, dt):
        self.game.car.update()
        if True:
            self.game.render()

if __name__ == "__main__":
    window = MyWindow(resizable=False)
    pyglet.clock.schedule_interval(window.update, 1/30.0)
    pyglet.app.run()