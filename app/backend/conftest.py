import os
import sys

# Make app/backend the root so `from encoders.fg_clip import ...` resolves.
sys.path.insert(0, os.path.dirname(__file__))
