from gym_robothor.envs import RoboThorEnv, env_generator
import sys
import numpy as np
import h5py
import click
import json
import pyglet

from PIL import Image

ALL_POSSIBLE_ACTIONS = [
	'MoveAhead',
	'MoveBack',
	'RotateRight',
	'RotateLeft',
	'Stop'   
]

class SimpleImageViewer(object):

  def __init__(self, display=None):
    self.window = None
    self.isopen = False
    self.display = display

  def imshow(self, arr):
    if self.window is None:
      height, width, channels = arr.shape
      self.window = pyglet.window.Window(width=width, height=height, display=self.display, caption="THOR Browser")
      self.width = width
      self.height = height
      self.isopen = True

    assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
    image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    image.blit(0,0)
    self.window.flip()

  def close(self):
    if self.isopen:
      self.window.close()
      self.isopen = False

  def __del__(self):
    self.close()

def run_single(file_name=None):
	# single env
	controller = RoboThorEnv()
	state = controller.reset()
	
	while True:  # making a loop
		try:  # used try so that if user pressed other than the given key error will not be shown
			key = click.getchar()
			if key =='a':  # Rotate Left
				state = controller.step(3)
			elif key =='d':
				state = controller.step(2)
			elif key =='w':
				state = controller.step(0)
			elif key =='s':
				state = controller.step(1)
			elif key =='z':
				state = controller.step(5)
			elif key =='x':
				state = controller.step(4)
			elif key =='c':
				state = controller.step(6)
			elif key =='v':
				controller.render()
			elif key =='q':
				controller.close()
				break
			elif key =='r':
				scene = input("Scene id: ")
				controller.controller.reset('FloorPlan{}'.format(scene))
			else:
				print("Key not supported! Try a, d, w, s, q, r.")
		except:
			print("Key not supported! Try a, d, w, s, q, r.")


def run_multi():
	for controller in env_generator('train_valid_', device='cpu'):
		controller.reset()
		while True:
			# print(controller.observation_space, controller.action_space.n)
			key = click.getchar()
			if key =='a':  # Rotate Left
				state, reward, done, _ = controller.step(0, return_event=True)
				if done:
					controller.reset()
			elif key =='d':
				state, reward, done, _  = controller.step(2, return_event=True)
				if done:
					controller.reset()
			elif key =='w':
				state, reward, done, _  = controller.step(1, return_event=True)
				if done:
					controller.reset()
			# elif key =='z':
			# 	state, reward, done, _  = controller.step(0, return_event=True)
			# 	if done:
			# 		controller.reset()
			# elif key =='x':
			# 	state, reward, done, _  = controller.step(5, return_event=True)
			# 	if done:
			# 		controller.reset()
			# elif key =='c':
			# 	state, reward, done, _  = controller.step(3, return_event=True)
			# 	if done:
			# 		controller.reset()
			elif key =='v':
				controller.render()
			elif key =='q':
				# controller.close()
				break
			elif key =='r':
				scene = input("Scene id: ")
				controller.controller.reset('FloorPlan{}'.format(scene))
			else:
				print("Key not supported! Try a, d, w, s, q, r.")

			# print(state)
			# print(_.metadata['collided'])
			# print(_.metadata['lastActionSuccess'])
			# print(_.metadata['agent'])




if __name__ == '__main__':
	
	run_multi()

	print("Goodbye.")