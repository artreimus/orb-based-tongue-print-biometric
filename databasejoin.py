import sqlite3
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import time
from datetime import datetime


#conn = sqlite3.connect('mainData.db')  # create connection with the database file with variable conn
#c = conn.cursor()  # cursor with variable c allows to execute sql commands with execute method

# Create Table -> uncomment the statements below when the database file is deleted to create a new one.
'''c.execute("""
     CREATE TABLE mainTable (
     person_id TEXT,
     first_name TEXT,
     last_name TEXT,
     image_name TEXT,
     image_data BLOB,
     descriptor TEXT)""")'''

conn = sqlite3.connect('mainData.db')  # create connection with the database file with variable conn
c = conn.cursor()  # cursor with variable c allows to execute sql commands with execute method
c.execute('ATTACH DATABASE "mainData2.db" as dbA')
c.execute('INSERT INTO mainTable SELECT * FROM dbA.mainTable')
conn.commit()



