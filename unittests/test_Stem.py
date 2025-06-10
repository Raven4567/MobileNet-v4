import unittest

import torch as t

import sys
import os
# This is the key part:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models # noqa: F401
import modules

device = t.device('cuda' if t.cuda.is_available() else 'cpu')