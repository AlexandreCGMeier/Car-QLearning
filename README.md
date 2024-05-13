# Car-QLearning

## UPDATE 12/05/2024

I've partially re-written the game now. It's rather simple. Game and Pyglet are nested, so it's difficult to get it to run on the server. We could probably introduce a mode where we don't do any graphics, but model the game as a purely 2D Matrix, where we have step, state, step, state, ... sequence. This way we could leverage strong server architecture. 
Pyglet is stuck at lower than v2.0 because of Shader Magic I don't want to implement.
From the AI's perspective the game is abstracted to step, reset and observe. So once the environment is defined, many architectures can be implemented very easily.

## How to install packages
In this folder activate venv, pip in venv. 

## How to instantiate VENV 
C:\Users\Alex\AppData\Local\Programs\Python\Python39\python.exe -m venv venv

## Powershell Policy Adaptation (start PS as Admin)
Set-ExecutionPolicy -ExecutionPolicy Unrestricted

## Update pip
C:\Users\Alex\Documents\GitHub\Car-QLearning\venv\bin\python.exe -m pip install --upgrade pip

## Requires Python 3.9 because of PyGame, that doesn't support 3.10 yet (14/05/22)

## Strategies

# 1: Plug in different DDQN algorithm.
Understand environment defined by CB (reward, punishment, episode, session cycle) and adapt. Now it's kind of a mess, study specification of a "good" environment and shape according to it.

# 2: Make it work
main.py at this points calculates something but doesn't render. QLearningFromMate.py renders but doesn't calculate. However, QLearningFromMate is the working approach as of now. Combine both to get a working thing. Potentially difficult because Game, AI and Render Logic is nested. 

IDEA Regarding 2: Make MyWindowClass in QLearningFromOldMate and see whether it still renders. If yes, then render issue from main.py might be easily resolvable.