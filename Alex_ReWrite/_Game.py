import numpy as np
import pygame
import pyglet
from Drawer import Drawer
from Globals import displayHeight, displayWidth
from PygameAdditionalMethods import *
from ShapeObjects import *

drawer = Drawer()
vec2 = pygame.math.Vector2


class Game:
    def __init__(self):
        trackImg = pyglet.image.load('images/track.png')
        self.trackSprite = pyglet.sprite.Sprite(trackImg, x=0, y=0)
        self.trackSprite.update(scale=0.6)
        self.car = Car()


    def render(self):
        glPushMatrix()
        #
        # glTranslatef(-1, -1, 0)
        # glScalef(1 / (displayWidth / 2), 1 / (displayHeight / 2), 1)

        # self.clear()
        self.trackSprite.draw()
        self.car.show()

        glPopMatrix()

class Car:
    def __init__(self):
        global vec2
        self.x = 174
        self.y = 141
        self.vel = 0
        self.direction = vec2(0, 1)
        self.direction = self.direction.rotate(180 / 12)
        self.acc = 0
        self.scalefactor = 0.8
        self.width = int(40*self.scalefactor)
        self.height = int(30*self.scalefactor)
        self.turningRate = 5.0 / self.width
        self.friction = 0.98
        self.maxSpeed = self.width / 4.0
        self.maxReverseSpeed = -1 * self.maxSpeed / 2.0
        self.accelerationSpeed = self.width / 160.0
        self.dead = False
        self.driftMomentum = 0
        self.driftFriction = 0.87
        self.lineCollisionPoints = []
        self.collisionLineDistances = []
        self.vectorLength = 300
        self.rewardAdditional = 0

        self.carPic = pyglet.image.load('images/car.png')
        self.carSprite = pyglet.sprite.Sprite(self.carPic, x=self.x, y=self.y)
        self.carSprite.update(rotation=0, scale_x=self.width / self.carSprite.width,
                              scale_y=self.height / self.carSprite.height)

        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False
        self.rewardNo = 0
        self.reward = 0

        self.score = 0
        self.lifespan = 0
    """
    draws the car to the screen
    """

    def reset(self):
        global vec2
        self.x = 174
        self.y = 141
        self.vel = 0
        self.direction = vec2(0, 1)
        self.direction = self.direction.rotate(180 / 12)
        self.acc = 0
        self.dead = False
        self.driftMomentum = 0
        self.lineCollisionPoints = []
        self.collisionLineDistances = []

        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False
        self.rewardNo = 0
        self.reward = 0
        self.rewardAdditional = 0

        self.lifespan = 0
        self.score = 0


    def show(self):
        # first calculate the center of the car in order to allow the
        # rotation of the car to be anchored around the center
        upVector = self.direction.rotate(90)
        drawX = self.direction.x * self.width / 2 + upVector.x * self.height / 2
        drawY = self.direction.y * self.width / 2 + upVector.y * self.height / 2
        self.carSprite.update(x=self.x - drawX, y=self.y - drawY, rotation=-get_angle(self.direction))
        self.carSprite.draw()

    """
     returns a vector of where a point on the car is after rotation 
     takes the position desired relative to the center of the car when the car is facing to the right
    """

    def getPositionOnCarRelativeToCenter(self, right, up):
        global vec2
        w = self.width
        h = self.height
        rightVector = vec2(self.direction)
        rightVector.normalize()
        upVector = self.direction.rotate(90)
        upVector.normalize()

        return vec2(self.x, self.y) + ((rightVector * right) + (upVector * up))

    def updateWithAction(self, actionNo):
        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False

        if actionNo == 0:
            self.turningLeft = True
        elif actionNo == 1:
            self.turningRight = True
        elif actionNo == 2:
            self.accelerating = True
        elif actionNo == 3:
            self.reversing = True
        elif actionNo == 4:
            self.accelerating = True
            self.turningLeft = True
        elif actionNo == 5:
            self.accelerating = True
            self.turningRight = True
        elif actionNo == 6:
            self.reversing = True
            self.turningLeft = True
        elif actionNo == 7:
            self.reversing = True
            self.turningRight = True
        elif actionNo == 8:
            pass
        totalReward = 0

        for _ in range(1):
            if not self.dead:
                self.lifespan+=1
                self.move()
                self.updateControls()

                if self.hitAWall():
                    self.dead = True
                    # return
                self.checkRewardGates()
                #print(self.rewardAdditional)
                #print(totalReward)
                #print(totalReward)
                #print("totalReward: " + str(totalReward))
                #print("totalAdditional Reward: " + str(self.rewardAdditional))
                #print("self.reward: " + str(self.reward))
                
        #print(self.reward)
        self.setVisionVectors()

        # self.update()

        self.reward += totalReward

    """
    called every frame
    """

    def update(self):
        self.move()


    def move(self):
        global vec2
        self.vel += self.acc
        self.vel *= self.friction
        self.constrainVel()

        driftVector = vec2(self.direction)
        driftVector = driftVector.rotate(90)

        addVector = vec2(0, 0)
        addVector.x += self.vel * self.direction.x
        addVector.x += self.driftMomentum * driftVector.x
        addVector.y += self.vel * self.direction.y
        addVector.y += self.driftMomentum * driftVector.y
        self.driftMomentum *= self.driftFriction

        if addVector.length() != 0:
            addVector.normalize()

        addVector.x * abs(self.vel)
        addVector.y * abs(self.vel)

        self.x += addVector.x
        self.y += addVector.y
