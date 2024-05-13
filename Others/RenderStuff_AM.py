from pygame import RESIZABLE
import pyglet

ball_image = pyglet.image.load('images/track.png')
ball = pyglet.sprite.Sprite(ball_image)
ball.update(scale=0.5)
#ball = ball_image

window = pyglet.window.Window(ball.width, ball.height)

def render():
    ball.draw()

@window.event
def on_draw():
    render()

pyglet.app.run()