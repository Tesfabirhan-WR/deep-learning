"""Common imports. In a notebook:  from dlkit.libs import *"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow info logs (set before importing tf)

#Dataframes
import pandas as pd

#numbers
import numpy as np
from operator import le
import random


#files and systems

import json
import sys
import warnings
from pathlib import Path

#plotting
import matplotlib.pyplot as plt
import seaborn as sns

#Machine learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input
import networkx as nx


__all__ = [
    "os", "json", "random", "sys", "warnings", "Path",
    "np", "pd", "plt", "tf", "keras", "layers",
    "train_test_split", "TfidfVectorizer", "LogisticRegression",
    "confusion_matrix", "classification_report", "accuracy_score",
]
