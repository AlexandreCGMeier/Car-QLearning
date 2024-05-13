import pyglet
from pyglet.gl import *
import pygame
import math
from pyglet.window import key
from Drawer import Drawer
# from PygameAdditionalMethods import *
from ShapeObjects import Line
import tensorflow as tf  # Deep Learning library
import numpy as np  # Handle matrices
from collections import deque
import random
import os
from Globals import displayHeight, displayWidth
from Game import Game

frameRate = 30.0

vec2 = pygame.math.Vector2

class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)
        # load background image
        self.game = Game()
        self.firstClick = True

    def on_draw(self):
        self.game.render()

    def on_mouse_press(self, x, y, button, modifiers):
        # print(x,y)
        if self.firstClick:
            self.clickPos = [x, y]
        else:
            pass
            #print("self.gates.append(RewardGate({}, {}, {}, {}))".format(self.clickPos[0],
             #                                                      displayHeight - self.clickPos[1],
              #                                                     x, displayHeight - y))
        #
            #print("self.gates.append(RewardGate({}, {}, {}, {}))".format(self.clickPos[0],
             #                                                      self.clickPos[1],
              #                                                     x, y))
        
        self.firstClick = not self.firstClick

    def update(self, dt):
        self.game.car.update()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.LEFT:
            self.game.make_action([1,0,0,0])
        if symbol == key.RIGHT:
            self.game.make_action([0,1,0,0])
        if symbol == key.UP:
            self.game.make_action([0,0,1,0])
        if symbol == key.DOWN:
            self.game.make_action(3[0,0,0,1])

    def on_key_release(self, symbol, modifiers):
        pass
    
if __name__ == "__main__":
    window = MyWindow(displayWidth, displayHeight, "AI Learns to Drive", resizable=False)
    pyglet.clock.schedule_interval(window.update, 1 / frameRate)
    pyglet.app.run()
