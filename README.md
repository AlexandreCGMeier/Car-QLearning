# Car-QLearning

## 07/11/24

Reward is now reset after every step instead of after every episode! Deep Q requires this. Previously, reward was Car-Property and increased cumulatively across episode. Deep Q only inputs state and that's it. Adjusted that. We might need to scale reward with decreasing distance towards RewardGate. Additionally, we increased range for "accepted" speeds which could lead to penalties. Exploration decay now scales with episode instead of step.

IT'S SOLVED. 6280 Episodes later and our guy flies.

## 21/05/24

Experiment 1 - Assess Environment Performance: Make every choice random (therefore disabling model) and output fps, take away rendering and output fps. Paperspace maxes out at 60fps using random choice and no rendering. Similar to XPS13. Rendering decreases this number from 60 to 45-ish. Random choice experiment can be by changing comments from "Render_QLearningOldMate.py".

Experiment 2 - Assess performance of device by taking given environment: Compare with "optimised environment" gum.env "Cartpole-v1" as used in minDQ implementation. We get 40fps on XPS13, but only 25 on paperspace. There seems to be something odd happening.

Conclusion from Exp1,2: Bottleneck for model performance is probably the reward structure of the environment, as opposed not enough training. Potential ideas are two-pronged: Either more thought through reward structure or more inputs for the model.

Reward Structure:

- Distance to Reward Gate and associated decrease (Integral over distance).
- Distance to Wall and penalising when falling below a threshold.

Streamlining State Description:

- Add new variables (Distance to Reward Gate)
- Remove redundant ones (Negative Velocity Vector)

GENERAL Idea: Make a manual driven mode, that outputs the reward of a run to detect bugs and "get a feel" for env.

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
