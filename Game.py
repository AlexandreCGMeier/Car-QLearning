import numpy as np
from Globals import *
from Drawer import Drawer
from ShapeObjects import *
from PygameAdditionalMethods import *
import pygame

drawer = Drawer()
vec2 = pygame.math.Vector2

class Game:
    no_of_actions = 9
    state_size = 20 #self.nbVect + 4

    def __init__(self):
        trackImg = pyglet.image.load('Track.png')
        self.trackSprite = pyglet.sprite.Sprite(trackImg, x=0, y=0)
        self.trackSprite.scale = displayHeight / trackImg.height

        # initiate walls
        self.walls = []
        self.gates = []

        self.set_walls()
        self.set_gates()
        self.firstClick = True

        self.car = Car(self.walls, self.gates)

    def set_walls(self):
        self.walls.append(Wall(149, 456, 142, 412))
        self.walls.append(Wall(143, 411, 145, 361))
        self.walls.append(Wall(146, 360, 149, 302))
        self.walls.append(Wall(150, 301, 165, 254))
        self.walls.append(Wall(166, 253, 188, 211))
        self.walls.append(Wall(189, 210, 210, 178))
        self.walls.append(Wall(211, 177, 260, 163))
        self.walls.append(Wall(261, 162, 337, 137))
        self.walls.append(Wall(338, 136, 381, 122))
        self.walls.append(Wall(382, 121, 507, 125))
        self.walls.append(Wall(508, 124, 620, 131))
        self.walls.append(Wall(621, 130, 822, 153))
        self.walls.append(Wall(823, 152, 845, 175))
        self.walls.append(Wall(846, 174, 859, 203))
        self.walls.append(Wall(860, 202, 854, 246))
        self.walls.append(Wall(855, 245, 835, 279))
        self.walls.append(Wall(836, 278, 808, 302))
        self.walls.append(Wall(809, 301, 733, 322))
        self.walls.append(Wall(734, 321, 531, 422))
        self.walls.append(Wall(532, 421, 775, 396))
        self.walls.append(Wall(776, 395, 891, 319))
        self.walls.append(Wall(892, 318, 936, 303))
        self.walls.append(Wall(937, 302, 1022, 292))
        self.walls.append(Wall(1023, 291, 1105, 324))
        self.walls.append(Wall(1106, 323, 1143, 353))
        self.walls.append(Wall(1144, 352, 1187, 434))
        self.walls.append(Wall(1188, 433, 1209, 491))
        self.walls.append(Wall(1210, 490, 1179, 617))
        self.walls.append(Wall(1180, 616, 1130, 647))
        self.walls.append(Wall(1131, 646, 1057, 675))
        self.walls.append(Wall(1058, 674, 246, 671))
        self.walls.append(Wall(247, 670, 190, 630))
        self.walls.append(Wall(191, 629, 160, 518))
        self.walls.append(Wall(161, 517, 143, 417))
        self.walls.append(Wall(144, 416, 144, 384))
        self.walls.append(Wall(207, 477, 199, 416))
        self.walls.append(Wall(200, 415, 196, 331))
        self.walls.append(Wall(197, 330, 204, 285))
        self.walls.append(Wall(205, 284, 251, 228))
        self.walls.append(Wall(252, 227, 308, 201))
        self.walls.append(Wall(309, 200, 394, 170))
        self.walls.append(Wall(395, 169, 771, 202))
        self.walls.append(Wall(772, 201, 779, 214))
        self.walls.append(Wall(780, 213, 774, 233))
        self.walls.append(Wall(775, 232, 729, 266))
        self.walls.append(Wall(730, 265, 473, 387))
        self.walls.append(Wall(474, 386, 429, 426))
        self.walls.append(Wall(430, 425, 424, 462))
        self.walls.append(Wall(425, 461, 432, 488))
        self.walls.append(Wall(433, 487, 492, 500))
        self.walls.append(Wall(493, 499, 796, 449))
        self.walls.append(Wall(797, 448, 923, 362))
        self.walls.append(Wall(924, 361, 1026, 350))
        self.walls.append(Wall(1027, 349, 1105, 379))
        self.walls.append(Wall(1106, 378, 1161, 494))
        self.walls.append(Wall(1162, 493, 1142, 571))
        self.walls.append(Wall(1143, 570, 1051, 623))
        self.walls.append(Wall(1052, 622, 287, 619))
        self.walls.append(Wall(288, 618, 237, 592))
        self.walls.append(Wall(238, 591, 222, 548))
        self.walls.append(Wall(223, 547, 196, 407))

    def set_gates(self):
        self.gates.append(RewardGate(195, 363, 144, 372))
        self.gates.append(RewardGate(199, 333, 145, 322))
        self.gates.append(RewardGate(203, 298, 162, 272))
        self.gates.append(RewardGate(235, 252, 183, 220))
        self.gates.append(RewardGate(283, 212, 258, 163))
        self.gates.append(RewardGate(338, 190, 314, 146))
        self.gates.append(RewardGate(382, 176, 364, 130))
        self.gates.append(RewardGate(452, 174, 458, 129))
        self.gates.append(RewardGate(557, 182, 564, 133))
        self.gates.append(RewardGate(666, 194, 674, 143))
        self.gates.append(RewardGate(751, 202, 769, 145))
        self.gates.append(RewardGate(778, 214, 832, 168))
        self.gates.append(RewardGate(773, 232, 844, 265))
        self.gates.append(RewardGate(742, 257, 778, 310))
        self.gates.append(RewardGate(674, 293, 703, 331))
        self.gates.append(RewardGate(627, 320, 645, 362))
        self.gates.append(RewardGate(569, 343, 593, 390))
        self.gates.append(RewardGate(523, 368, 548, 411))
        self.gates.append(RewardGate(520, 427, 453, 438))
        self.gates.append(RewardGate(517, 477, 542, 434))
        self.gates.append(RewardGate(589, 475, 583, 430))
        self.gates.append(RewardGate(661, 465, 654, 428))
        self.gates.append(RewardGate(738, 451, 733, 412))
        self.gates.append(RewardGate(797, 438, 776, 408))
        self.gates.append(RewardGate(854, 400, 824, 374))
        self.gates.append(RewardGate(894, 378, 861, 350))
        self.gates.append(RewardGate(926, 359, 909, 323))
        self.gates.append(RewardGate(962, 351, 958, 311))
        self.gates.append(RewardGate(1017, 345, 1018, 295))
        self.gates.append(RewardGate(1051, 352, 1061, 315))
        self.gates.append(RewardGate(1088, 371, 1111, 331))
        self.gates.append(RewardGate(1113, 397, 1153, 384))
        self.gates.append(RewardGate(1136, 440, 1176, 417))
        self.gates.append(RewardGate(1150, 476, 1199, 460))
        self.gates.append(RewardGate(1154, 522, 1198, 540))
        self.gates.append(RewardGate(1144, 557, 1191, 577))
        self.gates.append(RewardGate(1128, 586, 1157, 628))
        self.gates.append(RewardGate(1095, 602, 1109, 649))
        self.gates.append(RewardGate(1054, 628, 1057, 669))
        self.gates.append(RewardGate(1003, 629, 1002, 675))
        self.gates.append(RewardGate(924, 628, 924, 665))
        self.gates.append(RewardGate(835, 626, 832, 676))
        self.gates.append(RewardGate(768, 627, 766, 668))
        self.gates.append(RewardGate(699, 624, 700, 677))
        self.gates.append(RewardGate(607, 625, 605, 670))
        self.gates.append(RewardGate(539, 624, 536, 669))
        self.gates.append(RewardGate(480, 626, 479, 674))
        self.gates.append(RewardGate(429, 629, 420, 672))
        self.gates.append(RewardGate(379, 621, 374, 676))
        self.gates.append(RewardGate(339, 623, 324, 664))
        self.gates.append(RewardGate(290, 622, 242, 661))
        self.gates.append(RewardGate(260, 607, 199, 638))
        self.gates.append(RewardGate(231, 576, 187, 587))
        self.gates.append(RewardGate(221, 539, 162, 542))
        self.gates.append(RewardGate(213, 490, 158, 504))
        self.gates.append(RewardGate(204, 461, 151, 467))
        self.gates.append(RewardGate(199, 415, 144, 432))

    def new_episode(self):
        self.car.reset()

    def get_state(self):
        return self.car.getState()

    def make_action(self, action):
        # returns reward
        #actionNo = np.argmax(action)
        actionNo = int(np.argmax(action)) if isinstance(action, (list, np.ndarray)) else int(action)
        self.car.updateWithAction(actionNo)
        return self.car.reward

    def is_episode_finished(self):
        return self.car.dead

    def get_score(self):
        return self.car.score

    def get_lifespan(self):
        return self.car.lifespan

    def render(self):
        glPushMatrix()
        self.trackSprite.draw()

        # for w in self.walls:
        #     w.draw()
        for g in self.gates:
            g.draw()
        self.car.show()

        glPopMatrix()


class Wall:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = displayHeight - y1
        self.x2 = x2
        self.y2 = displayHeight - y2

        self.line = Line(self.x1, self.y1, self.x2, self.y2)
        self.line.setLineThinkness(3)
        self.line.setColor([255, 0, 0])

    """
    draw the line
    """
    def draw(self):
        self.line.draw()
    """
    returns true if the car object has hit this wall
    """

    def hitCar(self, car):
        global vec2
        cw = car.width
        ch = car.height
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
        self.y1 = displayHeight - y1
        self.x2 = x2
        self.y2 = displayHeight - y2
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
        ch = car.height
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



class Car:

    def __init__(self, walls, rewardGates):
        global vec2
        self.nbVect = 16
        self.angles = np.linspace(-180, 180, self.nbVect)
        self.x = 172
        self.y = 288
        self.vel = 0
        self.direction = vec2(0, 1)
        self.direction = self.direction.rotate(180 / 12)
        self.acc = 0
        self.width = 25
        self.height = 15
        self.turningRate = 5.0 / self.width
        self.friction = 0.98
        self.maxSpeed = self.width / 2.0
        self.maxReverseSpeed = self.maxSpeed / 16.0   #used as a minimum for speed
        self.accelerationSpeed = self.width / 160.0
        self.dead = False
        self.driftMomentum = 0
        self.driftFriction = 0.87
        self.lineCollisionPoints = []
        self.collisionLineDistances = []
        self.vectorLength = 600

        self.carPic = pyglet.image.load('Car.png')
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

        self.directionToRewardGate = self.rewardGates[self.rewardNo].center - vec2(self.x, self.y)

        self.reward = 0
        self.score = 0
        self.lifespan = 0
        self._delta_s = 0.0  # forward progress (pixels) toward the current gate during the last step
    """
    draws the car to the screen
    """

    def reset(self):
        global vec2
        self.x = 172
        self.y = 288
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
        self._delta_s = 0.0

        self.lifespan = 0
        self.score = 0
        for g in self.rewardGates:
            g.active = True

    def show(self):
        upVector = self.direction.rotate(90)
        drawX = self.direction.x * self.width / 2 + upVector.x * self.height / 2
        drawY = self.direction.y * self.width / 2 + upVector.y * self.height / 2
        self.carSprite.update(x=self.x - drawX, y=self.y - drawY, rotation=-get_angle(self.direction))
        self.carSprite.draw()
        # self.showCollisionVectors()

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
        #print("action number : " + str(actionNo))
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

        for i in range(1):
            if not self.dead:
                self.lifespan+=1

                self.updateControls()
                self.move()

                if self.hitAWall():
                    self.dead = True
                    #print("dead at x: " + str(self.x) + " y : " + str(displayHeight - self.y) + "u lived for : " + str(self.lifespan) + " reward : " + str(self.score))
                    # return
                self.checkRewardGates()
                totalReward += self.reward

        self.setVisionVectors()

        # self.update()

        self.reward = totalReward

    """
    called every frame
    """

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
        self.reward = -0.05
        if self.rewardGates[self.rewardNo].hitCar(self):
            self.rewardGates[self.rewardNo].active = False
            self.rewardNo += 1
            self.score += 1
            self.reward += 50
            if self.rewardNo == len(self.rewardGates):
                self.rewardNo = 0
                for g in self.rewardGates:
                    g.active = True
        # dense progress bonus ~ proportional to forward pixels this step
        self.reward += 0.02 * float(self._delta_s)   # tune 0.01..0.05 if needed
        # optional: clip for stability

        self.directionToRewardGate = self.rewardGates[self.rewardNo].center - vec2(self.x, self.y)

    """
    changes the position of the car to account for acceleration, velocity, friction and drift
    """

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
            addVector = addVector.normalize()

        # scale by speed (the previous lines didn't apply the scaling)
        step_vec = addVector * abs(self.vel)

        if 0 <= self.rewardNo < len(self.rewardGates):
            to_gate = self.rewardGates[self.rewardNo].center - vec2(self.x, self.y)
            if to_gate.length() > 1e-6 and step_vec.length() > 0:
                t_hat = to_gate.normalize()
                # count only *forward* progress toward the gate (never backward)
                self._delta_s = max(0.0, step_vec.x * t_hat.x + step_vec.y * t_hat.y)
            else:
                self._delta_s = 0.0
        else:   
            self._delta_s = 0.0

        # apply movement
        self.x += step_vec.x
        self.y += step_vec.y

    """
    keeps the velocity of the car within the maximum and minimum speeds
    """

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
                self.acc = -2 * self.accelerationSpeed
            else:
                self.acc = 0
                self.vel = 0

    """
    checks every wall and if the car has hit a wall returns true    
    """

    def hitAWall(self):
        for wall in self.walls:
            if wall.hitCar(self):
                return True
        return False

    """
    returns the point of collision of a line (x1,y1,x2,y2) with the walls, 
    if multiple walls are hit it returns the closest collision point
    """

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

        normalizedForwardVelocity = max(0, (self.vel-self.maxReverseSpeed) / (self.maxSpeed-self.maxReverseSpeed))
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

        normalizedState = [*normalizedVisionVectors, normalizedForwardVelocity,
                           normalizedPosDrift, normalizedNegDrift, normalizedAngleOfNextGate]
        return np.array(normalizedState)

    def setVisionVectors(self):
        self.collisionLineDistances = []
        self.lineCollisionPoints = []
        for i in self.angles:
            self.setVisionVector(0, 0, i)

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