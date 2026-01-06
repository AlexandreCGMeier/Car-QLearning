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
    no_of_actions = 9
    state_size = 15
    def __init__(self):
        trackImg = pyglet.image.load('track.png')
        self.trackSprite = pyglet.sprite.Sprite(trackImg, x=0, y=0)
        self.trackSprite.update(scale=0.6)
        self.walls = []
        self.gates = []
        self.set_walls()
        self.set_gates()
        self.firstClick = True
        self.car = Car(self.walls, self.gates)

    def set_walls(self):
        self.walls.append(Wall(136, 461, 116, 115))
        self.walls.append(Wall(116, 115, 196, 42))
        self.walls.append(Wall(196, 42, 324, 50))
        self.walls.append(Wall(324, 50, 332, 250))
        self.walls.append(Wall(333, 251, 394, 255))
        self.walls.append(Wall(394, 255, 421, 49))
        self.walls.append(Wall(421, 49, 753, 40))
        self.walls.append(Wall(753, 40, 779, 163))
        self.walls.append(Wall(779, 163, 549, 178))
        self.walls.append(Wall(549, 178, 538, 240))
        self.walls.append(Wall(538, 240, 833, 239))
        self.walls.append(Wall(833, 239, 817, 459))
        self.walls.append(Wall(817, 459, 487, 480))
        self.walls.append(Wall(487, 480, 478, 443))
        self.walls.append(Wall(478, 443, 448, 406))
        self.walls.append(Wall(448, 406, 399, 389))
        self.walls.append(Wall(399, 389, 344, 406))
        self.walls.append(Wall(344, 406, 304, 432))
        self.walls.append(Wall(304, 432, 280, 470))
        self.walls.append(Wall(280, 470, 143, 474))
        self.walls.append(Wall(143, 474, 122, 394))
        self.walls.append(Wall(174, 433, 169, 117))
        self.walls.append(Wall(169, 117, 226, 85))
        self.walls.append(Wall(229, 85, 277, 89))
        self.walls.append(Wall(276, 90, 290, 287))
        self.walls.append(Wall(290, 287, 420, 297))
        self.walls.append(Wall(420, 297, 448, 242))
        self.walls.append(Wall(448, 242, 464, 98))
        self.walls.append(Wall(464, 98, 558, 80))
        self.walls.append(Wall(556, 80, 716, 94))
        self.walls.append(Wall(716, 94, 722, 130))
        self.walls.append(Wall(722, 130, 533, 133))
        self.walls.append(Wall(533, 133, 500, 161))
        self.walls.append(Wall(500, 161, 494, 269))
        self.walls.append(Wall(494, 269, 537, 286))
        self.walls.append(Wall(537, 286, 780, 284))
        self.walls.append(Wall(780, 284, 795, 309))
        self.walls.append(Wall(795, 309, 784, 419))
        self.walls.append(Wall(784, 419, 522, 424))
        self.walls.append(Wall(522, 424, 498, 380))
        self.walls.append(Wall(498, 380, 473, 352))
        self.walls.append(Wall(473, 352, 399, 328))
        self.walls.append(Wall(399, 328, 340, 344))
        self.walls.append(Wall(340, 344, 289, 364))
        self.walls.append(Wall(289, 364, 263, 394))
        self.walls.append(Wall(263, 394, 246, 425))
        self.walls.append(Wall(246, 425, 181, 427))
        self.walls.append(Wall(181, 427, 167, 312))
    def set_gates(self):
        self.gates.append(RewardGate(121, 391, 169, 394))
        self.gates.append(RewardGate(120, 420, 167, 423))
        self.gates.append(RewardGate(118, 437, 170, 439))
        self.gates.append(RewardGate(169, 458, 118, 463))
        self.gates.append(RewardGate(116, 488, 168, 477))
        self.gates.append(RewardGate(181, 487, 155, 516))
        self.gates.append(RewardGate(169, 529, 195, 499))
        self.gates.append(RewardGate(213, 508, 180, 543))
        self.gates.append(RewardGate(211, 555, 222, 517))
        self.gates.append(RewardGate(254, 513, 259, 554))
        self.gates.append(RewardGate(284, 550, 273, 517))
        self.gates.append(RewardGate(276, 508, 321, 526))
        self.gates.append(RewardGate(325, 499, 279, 491))
        self.gates.append(RewardGate(278, 470, 325, 476))
        self.gates.append(RewardGate(329, 458, 282, 455))
        self.gates.append(RewardGate(281, 434, 327, 440))
        self.gates.append(RewardGate(327, 422, 282, 421))
        self.gates.append(RewardGate(282, 397, 328, 406))
        self.gates.append(RewardGate(332, 379, 288, 374))
        self.gates.append(RewardGate(287, 345, 329, 355))
        self.gates.append(RewardGate(333, 347, 307, 312))
        self.gates.append(RewardGate(328, 310, 353, 346))
        self.gates.append(RewardGate(363, 345, 363, 311))
        self.gates.append(RewardGate(383, 307, 382, 345))
        self.gates.append(RewardGate(392, 344, 419, 312))
        self.gates.append(RewardGate(429, 325, 398, 358))
        self.gates.append(RewardGate(398, 371, 447, 361))
        self.gates.append(RewardGate(448, 382, 398, 389))
        self.gates.append(RewardGate(401, 415, 449, 409))
        self.gates.append(RewardGate(454, 428, 404, 434))
        self.gates.append(RewardGate(407, 454, 453, 448))
        self.gates.append(RewardGate(457, 467, 412, 470))
        self.gates.append(RewardGate(408, 495, 459, 480))
        self.gates.append(RewardGate(463, 496, 415, 512))
        self.gates.append(RewardGate(419, 537, 466, 500))
        self.gates.append(RewardGate(477, 505, 464, 551))
        self.gates.append(RewardGate(496, 550, 511, 514))
        self.gates.append(RewardGate(530, 517, 528, 552))
        self.gates.append(RewardGate(552, 550, 555, 521))
        self.gates.append(RewardGate(583, 518, 582, 551))
        self.gates.append(RewardGate(606, 552, 612, 519))
        self.gates.append(RewardGate(640, 515, 641, 555))
        self.gates.append(RewardGate(667, 555, 668, 513))
        self.gates.append(RewardGate(692, 511, 692, 555))
        self.gates.append(RewardGate(726, 557, 712, 512))
        self.gates.append(RewardGate(715, 500, 750, 530))
        self.gates.append(RewardGate(763, 511, 719, 491))
        self.gates.append(RewardGate(721, 478, 770, 480))
        self.gates.append(RewardGate(774, 451, 722, 467))
        self.gates.append(RewardGate(735, 434, 705, 472))
        self.gates.append(RewardGate(679, 470, 685, 435))
        self.gates.append(RewardGate(655, 433, 640, 470))
        self.gates.append(RewardGate(599, 471, 607, 429))
        self.gates.append(RewardGate(577, 425, 571, 464))
        self.gates.append(RewardGate(523, 465, 556, 431))
        self.gates.append(RewardGate(545, 408, 498, 424))
        self.gates.append(RewardGate(496, 402, 545, 398))
        self.gates.append(RewardGate(541, 377, 494, 378))
        self.gates.append(RewardGate(540, 367, 498, 343))
        self.gates.append(RewardGate(523, 319, 565, 359))
        self.gates.append(RewardGate(584, 359, 567, 316))
        self.gates.append(RewardGate(598, 315, 604, 358))
        self.gates.append(RewardGate(629, 358, 625, 320))
        self.gates.append(RewardGate(655, 317, 655, 358))
        self.gates.append(RewardGate(694, 358, 692, 311))
        self.gates.append(RewardGate(722, 312, 720, 356))
        self.gates.append(RewardGate(745, 360, 745, 319))
        self.gates.append(RewardGate(769, 314, 771, 359))
        self.gates.append(RewardGate(802, 360, 785, 311))
        self.gates.append(RewardGate(788, 303, 828, 334))
        self.gates.append(RewardGate(832, 316, 793, 296))
        self.gates.append(RewardGate(793, 281, 825, 281))
        self.gates.append(RewardGate(825, 266, 792, 265))
        self.gates.append(RewardGate(789, 249, 823, 251))
        self.gates.append(RewardGate(821, 233, 787, 237))
        self.gates.append(RewardGate(786, 215, 824, 219))
        self.gates.append(RewardGate(822, 197, 788, 204))
        self.gates.append(RewardGate(784, 187, 818, 184))
        self.gates.append(RewardGate(818, 156, 786, 185))
        self.gates.append(RewardGate(776, 182, 794, 141))
        self.gates.append(RewardGate(771, 138, 760, 182))
        self.gates.append(RewardGate(734, 184, 736, 140))
        self.gates.append(RewardGate(708, 137, 704, 173))
        self.gates.append(RewardGate(663, 180, 670, 134))
        self.gates.append(RewardGate(640, 129, 636, 174))
        self.gates.append(RewardGate(596, 177, 597, 129))
        self.gates.append(RewardGate(558, 125, 554, 178))
        self.gates.append(RewardGate(531, 177, 510, 122))
        self.gates.append(RewardGate(482, 134, 514, 186))
        self.gates.append(RewardGate(509, 200, 466, 168))
        self.gates.append(RewardGate(456, 186, 497, 215))
        self.gates.append(RewardGate(485, 237, 442, 194))
        self.gates.append(RewardGate(430, 203, 443, 252))
        self.gates.append(RewardGate(405, 267, 399, 213))
        self.gates.append(RewardGate(383, 207, 353, 261))
        self.gates.append(RewardGate(332, 255, 349, 196))
        self.gates.append(RewardGate(322, 185, 278, 223))
        self.gates.append(RewardGate(262, 200, 300, 172))
        self.gates.append(RewardGate(293, 157, 258, 195))
        self.gates.append(RewardGate(249, 179, 279, 141))
        self.gates.append(RewardGate(255, 133, 240, 175))
        self.gates.append(RewardGate(218, 176, 206, 129))
        self.gates.append(RewardGate(179, 127, 192, 171))
        self.gates.append(RewardGate(178, 174, 138, 147))
        self.gates.append(RewardGate(134, 174, 181, 190))
        self.gates.append(RewardGate(178, 218, 131, 216))
        self.gates.append(RewardGate(129, 242, 180, 246))
        self.gates.append(RewardGate(174, 285, 125, 289))
        self.gates.append(RewardGate(125, 309, 168, 305))
        self.gates.append(RewardGate(172, 341, 122, 343))
        self.gates.append(RewardGate(123, 369, 168, 367))

    def new_episode(self):
        self.car.reset()

    def get_state(self):
        return self.car.getState()
        
    def make_action(self, action):
        actionNo = np.argmax(action)
        self.car.updateWithAction(actionNo)
        currReward = self.car.reward
        self.car.reward = 0
        return currReward

    def is_episode_finished(self):
        return self.car.dead

    def get_score(self):
        return self.car.score

    def get_lifespan(self):
        return self.car.lifespan

    def render(self):
        glPushMatrix()
        self.trackSprite.draw()
        self.car.show()
        for w in self.walls:
            w.draw()
        for g in self.gates:
            g.draw()
        glPopMatrix()

class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = displayHeight - y1
        self.x2 = x2
        self.y2 = displayHeight - y2
        self.line = Line(self.x1, self.y1, self.x2, self.y2)
        self.line.setLineThinkness(2)

    def draw(self):
        self.line.draw()

    def hitCar(self, car):
        global vec2
        cw = car.width
        # since the car sprite isn't perfectly square the hitbox is a little smaller than the width of the car
        ch = car.height - 4
        rightVector = vec2(car.direction)
        upVector = vec2(car.direction).rotate(-90)
        carCorners = []
        cornerMultipliers = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        carPos = vec2(car.x, car.y)
        for i in range(4):
            carCorners.append(carPos + (rightVector * cw / 2 * cornerMultipliers[i][0]) +
                              (upVector * ch / 2 * cornerMultipliers[i][1]))
        for i in range(4):
            j = i + 1
            j = j % 4
            if linesCollided(self.x1, self.y1, self.x2, self.y2, carCorners[i].x, carCorners[i].y, carCorners[j].x,
                             carCorners[j].y):
                return True
        return False

class RewardGate:
    def __init__(self, x1, y1, x2, y2):
        global vec2
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.active = True
        self.line = Line(self.x1, self.y1, self.x2, self.y2)
        self.line.setLineThinkness(1)
        self.line.setColor([0, 255, 0])
        self.center = vec2((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    """
    draw the line
    """

    def draw(self):
        if self.active:
            self.line.draw()
    """
    returns true if the car object has hit this wall
    """
    def hitCar(self, car):
        if not self.active:
            return False
        global vec2
        cw = car.width
        # since the car sprite isn't perfectly square the hitbox is a little smaller than the width of the car
        ch = car.height - 4
        rightVector = vec2(car.direction)
        upVector = vec2(car.direction).rotate(-90)
        carCorners = []
        cornerMultipliers = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        carPos = vec2(car.x, car.y)
        #print("Car Pos x/y: " + str(car.x) + str(car.y))
        #print("Next Reward Gate: " + str(self.x1) + str(self.y1))
        for i in range(4):
            carCorners.append(carPos + (rightVector * cw / 2 * cornerMultipliers[i][0]) +
                              (upVector * ch / 2 * cornerMultipliers[i][1]))

        for i in range(4):
            j = i + 1
            j = j % 4
            if linesCollided(self.x1, self.y1, self.x2, self.y2, carCorners[i].x, carCorners[i].y, carCorners[j].x,
                             carCorners[j].y):
                return True
        return False

class Car:
    def __init__(self, walls, rewardGates):
        global vec2
        self.x = 152
        self.y = 334
        self.vel = 0
        self.direction = vec2(0, 1)
        self.direction = self.direction.rotate(180 / 12)
        self.acc = 0
        self.scalefactor = 0.2
        self.width = int(40*self.scalefactor)
        self.height = int(30*self.scalefactor)
        self.turningRate = 3.0 / self.width
        self.friction = 0.98
        self.maxSpeed = self.width / 4.0
        self.maxReverseSpeed = -1 * self.maxSpeed / 2.0
        self.accelerationSpeed = 0.1#self.width / 160.0
        self.dead = False
        self.driftMomentum = 0
        self.driftFriction = 0.87
        self.lineCollisionPoints = []
        self.collisionLineDistances = []
        self.vectorLength = 300
        self.rewardAdditional = 0

        self.carPic = pyglet.image.load('car.png')
        self.carSprite = pyglet.sprite.Sprite(self.carPic, x=self.x, y=self.y)
        self.carSprite.update(rotation=0, scale_x=self.width / self.carSprite.width,
                              scale_y=self.height / self.carSprite.height)

        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False
        self.walls = walls
        self.rewardGates = rewardGates
        self.rewardNo = 0
        self.rewardGates[self.rewardNo].line.setColor([255, 0, 0])
        self.directionToRewardGate = self.rewardGates[self.rewardNo].center - vec2(self.x, self.y)
        #print("X Coordinate of first Gate:" + str(self.rewardGates[self.rewardNo].y1))
        self.reward = 0
        self.score = 0
        self.lifespan = 0

    def reset(self):
        global vec2
        self.x = 152
        self.y = 334
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
        for g in self.rewardGates:
            g.active = True
            g.line.setColor([0, 255, 0])
        self.rewardGates[self.rewardNo].line.setColor([255, 0, 0])

    def show(self):
        # first calculate the center of the car in order to allow the
        # rotation of the car to be anchored around the center
        upVector = self.direction.rotate(90)
        drawX = self.direction.x * self.width / 2 + upVector.x * self.height / 2
        drawY = self.direction.y * self.width / 2 + upVector.y * self.height / 2
        self.carSprite.update(x=self.x - drawX, y=self.y - drawY, rotation=-get_angle(self.direction))
        self.carSprite.draw()
        self.showCollisionVectors()

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
    
        if not self.dead:
            self.lifespan+=1
            self.reward += 2
            self.move()
            self.updateControls()
            if self.hitAWall():
                self.dead = True
                # return
            self.checkRewardGates()
        self.setVisionVectors()
        #self.reward += totalReward

    def update(self):
        if not self.dead:
            self.updateControls()
            self.move()
            if self.hitAWall():
                self.dead = True
                # return
            self.checkRewardGates()
            self.setVisionVectors()

    def checkRewardGates(self):
        global vec2
        self.rewardPenalty = 0
        self.rewardAdditional = 0

        if self.vel < 0.1:
            self.rewardPenalty = -3
        if self.rewardGates[self.rewardNo].hitCar(self):
            self.rewardGates[self.rewardNo].active = False
            self.rewardNo += 1
            #print(self.rewardNo)
            if self.rewardNo >= len(self.rewardGates):
                self.rewardNo = 0
                for g in self.rewardGates:
                    g.active = True
                    g.line.setColor([0, 255, 0])
            self.rewardGates[self.rewardNo].line.setColor([255, 0, 0])
            self.score += 1
            self.rewardAdditional += 1000
            if self.rewardNo == len(self.rewardGates):
                self.rewardNo = 0
                for g in self.rewardGates:
                    g.active = True
        self.reward = self.reward + self.rewardAdditional + self.rewardPenalty
        self.directionToRewardGate = self.rewardGates[self.rewardNo].center - vec2(self.x, self.y)

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

    def constrainVel(self):
        if self.maxSpeed < self.vel:
            self.vel = self.maxSpeed
        elif self.vel < self.maxReverseSpeed:
            self.vel = self.maxReverseSpeed

    def updateControls(self):
        multiplier = 1
        if abs(self.vel) < 5:
            multiplier = abs(self.vel) / 5
        if self.vel < 0:
            multiplier *= -1

        driftAmount = self.vel * self.turningRate * self.width / (9.0 * 8.0)
        if self.vel < 5:
            driftAmount = 0

        if self.turningLeft:
            self.direction = self.direction.rotate(radiansToAngle(self.turningRate) * multiplier)
            self.driftMomentum -= driftAmount

        elif self.turningRight:
            self.direction = self.direction.rotate(-radiansToAngle(self.turningRate) * multiplier)
            self.driftMomentum += driftAmount
        self.acc = 0

        if self.accelerating:
            if self.vel < 0:
                self.acc = 3 * self.accelerationSpeed
            else:
                self.acc = self.accelerationSpeed

        elif self.reversing:
            if self.vel > 0:
                self.acc = -3 * self.accelerationSpeed
            else:
                self.acc = -1 * self.accelerationSpeed

    def hitAWall(self):
        for wall in self.walls:
            if wall.hitCar(self):
                return True
        return False

    def getCollisionPointOfClosestWall(self, x1, y1, x2, y2):
        global vec2
        minDist = 2 * displayWidth
        closestCollisionPoint = vec2(0, 0)
        for wall in self.walls:
            collisionPoint = getCollisionPoint(x1, y1, x2, y2, wall.x1, wall.y1, wall.x2, wall.y2)
            if collisionPoint is None:
                continue
            if dist(x1, y1, collisionPoint.x, collisionPoint.y) < minDist:
                minDist = dist(x1, y1, collisionPoint.x, collisionPoint.y)
                closestCollisionPoint = vec2(collisionPoint)
        return closestCollisionPoint

    def getState(self):
        self.setVisionVectors()
        
        normalizedVisionVectors = [1 - (max(1.0, line) / self.vectorLength) for line in self.collisionLineDistances]
        normalizedForwardVelocity = max(0.0, self.vel / self.maxSpeed)
        normalizedReverseVelocity = max(0.0, self.vel / self.maxReverseSpeed)
        if self.driftMomentum > 0:
            normalizedPosDrift = self.driftMomentum / 5
            normalizedNegDrift = 0
        else:
            normalizedPosDrift = 0
            normalizedNegDrift = self.driftMomentum / -5
        normalizedAngleOfNextGate = (get_angle(self.direction) - get_angle(self.directionToRewardGate)) % 360
        if normalizedAngleOfNextGate > 180:
            normalizedAngleOfNextGate = -1 * (360 - normalizedAngleOfNextGate)
        normalizedAngleOfNextGate /= 180

        normalizedState = [*normalizedVisionVectors, normalizedForwardVelocity, normalizedReverseVelocity,
                           normalizedPosDrift, normalizedNegDrift, normalizedAngleOfNextGate]
        return np.array(normalizedState)

    def setVisionVectors(self):
        h = self.height - 4
        w = self.width
        self.collisionLineDistances = []
        self.lineCollisionPoints = []
        self.setVisionVector(w / 2, 0, 0)
        self.setVisionVector(w / 2, -h / 2, -180 / 16)
        self.setVisionVector(w / 2, -h / 2, -180 / 4)
        self.setVisionVector(w / 2, -h / 2, -4 * 180 / 8)
        self.setVisionVector(w / 2, h / 2, 180 / 16)
        self.setVisionVector(w / 2, h / 2, 180 / 4)
        self.setVisionVector(w / 2, h / 2, 4 * 180 / 8)
        self.setVisionVector(-w / 2, -h / 2, -6 * 180 / 8)
        self.setVisionVector(-w / 2, h / 2, 6 * 180 / 8)
        self.setVisionVector(-w / 2, 0, 180)

    def setVisionVector(self, startX, startY, angle):
        collisionVectorDirection = self.direction.rotate(angle)
        collisionVectorDirection = collisionVectorDirection.normalize() * self.vectorLength
        startingPoint = self.getPositionOnCarRelativeToCenter(startX, startY)
        collisionPoint = self.getCollisionPointOfClosestWall(startingPoint.x, startingPoint.y,
                                                             startingPoint.x + collisionVectorDirection.x,
                                                             startingPoint.y + collisionVectorDirection.y)
        if collisionPoint.x == 0 and collisionPoint.y == 0:
            self.collisionLineDistances.append(self.vectorLength)
        else:
            self.collisionLineDistances.append(
                dist(startingPoint.x, startingPoint.y, collisionPoint.x, collisionPoint.y))
        self.lineCollisionPoints.append(collisionPoint)

    def showCollisionVectors(self):
        global drawer
        for point in self.lineCollisionPoints:
            drawer.setColor([255, 0, 0])
            drawer.circle(point.x, point.y, 5)