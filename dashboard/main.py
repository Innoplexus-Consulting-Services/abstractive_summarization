import os
import sys
import kivy

import numpy as np

from kivy.app import App
from kivy.uix.button import Label
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.base import runTouchApp

class RootWidget(BoxLayout):

	def __init__(self, **kwargs):
		super(RootWidget, self).__init__(**kwargs)
		Window.size = (1366, 768)


class textSummarizationApp(App):
    def build(self):
        return RootWidget()

if __name__ == '__main__':
    textSummarizationApp().run()
