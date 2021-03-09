#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:12:37 2020

@author: haubnerj
"""

def save_obj(obj, name):
    with open('obj/'+name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open('obj/'+name +'.pkl', 'rb') as f:
        return pickle.load(f)