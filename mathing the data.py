# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:29:36 2018

@author: Jack
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

v_ranges = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]

path = r"C:\Users\Jack\Documents\Uni\GDP\Picoscope\Low Gear\36.01V, 0.48A, 2.67V\36.01V, 0.48A, 2.67V_01.txt"

def open_file(path, us=True):
    try:
        path
    except SyntaxError:
        path = "%r" % path # stops the character escapes from happening
    file = np.loadtxt(path, skiprows=2)
    file[:, 0] -= file[0,0]
    f = open(path)
    f.readline()
    line = f.readline()
    if line.split()[0] == "(ms)":
        if us is False:
            file[:, 0] = file[:, 0]/1000
        else:
            file[:, 0] *= 1000
            file[:, 0] = (file[:, 0].round()).astype(int)
        return file
    elif line.split()[0] == "(us)":
        if us is False:
            file[:, 0] = file[:, 0]/1000000
        else:
            file[:, 0] = (file[:, 0].round()).astype(int)
        return file
    elif line.split()[0] == "(s)":
        if us is True:
            file[:, 0] *= 1000000
            file[:, 0] = (file[:, 0].round()).astype(int)
        return file
    else:
        print("Time value not known")
        return line


def find_diff(res):
    k1 = res[:-1]
    return res[1:]-k1

def level_shift(data, threshold=10000, plot=False, us=None):
    """finds the pwm frequency and duty cycle. This one works by subtracting
    a low-pass filtered set of results from the results, so that the on states
    are above zero and the off states are below 0. Using the low-passed results
    means that the level shift on a specific point is relative its magnitude.
    The results are then converted to binary, and the rising and falling edges
    are found."""
    time = data[:, 0]
    if us is None:
        if type(time[0]) == int:
            us = True
        elif time[1]-time[0] >= 1:
            us = True
        else:
            us = False
    if us is True:
        threshold *= (1/1000000)
    print(threshold)
    val = data[:, 1]
    N = len(val)
    fs = 1/np.average(np.abs(find_diff(time)))
    f_max = fs/2
    if N & 0x1:
        f_max *= (N-1)/N
    #val_bin = val>=threshold
    freqs = np.linspace(0, f_max, num=1+ np.int(np.round(N/2)))
    mags = np.abs(np.fft.rfft(val))
    #plt.semilogx(freqs, mags)
    #plt.plot(freqs[1:], mags[1:])
    w = 2*np.pi*freqs[1+np.argmax(mags[1:]*(freqs[1:]<threshold))] # angular velocity of the wheel
    pwm = freqs[1+np.argmax(mags[1:]*(freqs[1:]>threshold))]
    cutoff = 3*np.round(w)*2/1000000
    b, a = signal.butter(1, cutoff, btype="low")
    res = val - signal.lfilter(b, a, val)*2/3   # level-shift proprtional to magnitude
    res = ((res*(res>1))>1)*np.ones(len(res)) # convert res to binary
    res = find_diff(res)  # find edges, preserving edge direction
    if plot is True:
        plt.plot(time[:50000], res[:50000])
        plt.plot(time[:50000], val[:50000])
    time=time[:-1]*res          # keep only the times that have an edge
    time=time[np.nonzero(time)] # get rid of the elements that are 0
    #plot_time = np.trim_zeros(time*(np.abs(time)<(50/1000)))
    #plt.scatter(np.abs(plot_time), np.ones(len(plot_time))*30)
    #plt.plot(time)
    rising = time*(time>0)
    rising=rising[np.nonzero(rising)]
    falling = time*(time<0)
    falling=np.abs(falling[np.nonzero(falling)])
    periods = np.append(find_diff(rising), find_diff(falling))
    periods = periods*(periods<(1/threshold))
    periods = periods[np.nonzero(periods)]
    #return periods
    if time[0]>0:
        print(rising[0], falling[0])
        t_on = falling - rising
        t_off = falling[1:]-rising[:-1]
    else:
        print("f")
        t_on = falling[1:]-rising[:-1]
        t_off = falling-rising
    t_on = t_on*(t_on<(1/threshold))
    t_on = t_on[np.nonzero(t_on)]
    t_off = t_off*(t_off<(1/threshold))
    t_off = t_off[np.nonzero(t_off)]
    return periods, t_on, t_off
    
    
def test_filt(fc=30000, D=0.8, res=10, N=10, f=15000):
    m = 2**res
    SR = f * m
    on = np.ones(np.int(np.round(m*D)))
    off = np.zeros(m-len(on))
    tot = np.append(on, off)-0.5
    vals = np.zeros(N*m)
    c = 0
    while c< N:
        vals[c*m:(c+1)*m] = tot
        c+=1
    #b = np.array([-np.exp(-fc*2*np.pi/f)])
    #a = np.array([-1])
    b, a = signal.butter(2, fc*2/SR, btype="high")
    times = np.linspace(0, N/f, len(vals))
    #vals = np.array([times, vals])
    res = signal.lfilter(b, a, vals)
    plt.plot(times, vals)
    plt.plot(times, res)
    return res, vals
    