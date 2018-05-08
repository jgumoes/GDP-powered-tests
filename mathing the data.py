# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 11:29:36 2018

@author: Jack
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from numba import jit


path = r"C:\Users\Jack\Documents\Uni\GDP\Picoscope\Low Gear\36.01V, 0.48A, 2.67V\36.01V, 0.48A, 2.67V_01.txt"

out_path="Low Gear.txt"
out_file = None

#@jit
def do_all(folder_list):
    """does all of the results in a folder. folder_list needs be a list of 
    paths that the results are in"""
    folder_list = sorted(folder_list)
    check = input("the output filepath is %s, \nContinue?" % out_path)
    if check[0] == "y":
        pass
    else:
        return "Cancelled"
    if out_path is not None:
        global out_file
        out_file = open(out_path, 'w')
    for i in folder_list:
        open_dir(i)
    out_file.close()
    out_file = None

#@jit
def open_dir(path, us=True):
    """process all the data files in a directory"""
    files = sorted(os.listdir(path))
    if path.split("\\")[-1] == "":
        pass
    else:
        path += "\\"
    periods = np.array([])
    t_on = np.array([])
    t_off = np.array([])
    ws = np.array([])
    for i in files:
        data = open_file(path+i, us=us)
        out = level_shift(data, plot=False)
        periods = np.append(periods, out[0])
        t_on = np.append(t_on, out[1])
        t_off = np.append(t_off, out[2])
        if (len(ws) != 0):
            if (np.abs((out[3]/np.average(ws))-1)<=1):
                ws = np.append(ws, out[3])
        else:
            ws = np.append(ws, out[3])
        #print(periods, t_on, t_off)
    print(path.split("\\")[-2])
    p_stats = stats(periods)
    print("Periods:\t", p_stats)
    t_on_stats = stats(t_on)
    print("On times:\t", t_on_stats)
    t_off_stats = stats(t_off)
    print("Off times:\t", t_off_stats)
    w = np.average(ws)
    print("Angular Velocity:\t", w)
    if out_file is not None:
        line = ""
        for i in path.split("\\")[-2].split(", "):
            line += i[:-1] + ", "
        line += str(w)+", "
        line+="Periods:, "+np.array2string(p_stats, separator=",").replace('\n', '')[1:-1]+", "
        line+="On_times:, "+np.array2string(t_on_stats, separator=",").replace('\n', '')[1:-1]+", "
        line+="Off_times:, "+np.array2string(t_off_stats, separator=",").replace('\n', '')[1:-1]+"\n"
        out_file.write(line)

@jit
def open_file(path, us=True):
    #try:
    #    path
    #except SyntaxError:
    #    path = "%r" % path # stops the character escapes from happening
    file = np.loadtxt(path, skiprows=2)
    file[:, 0] -= file[0,0]
    f = open(path)
    f.readline()
    line = f.readline()
    f.close()
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

#@jit
def find_diff(res):
    k1 = res[:-1]
    return res[1:]-k1

#@jit
def level_shift(data, threshold=10000, plot=False, us=None):
    """finds the pwm frequency and duty cycle. This one works by subtracting
    a low-pass filtered set of results from the results, so that the on states
    are above zero and the off states are below 0. Using the low-passed results
    means that the level shift on a specific point is relative its magnitude.
    The results are then converted to binary, and the rising and falling edges
    are found."""
    if (np.shape(data)[0] == 2) and (np.shape(data)[1] != 2):
        data = np.transpose(data)
    time = data[:, 0]
    if us is None:
        if (type(time[0]) == int) or (time[1]-time[0] >= 1):
            us = True
            T_min = 10**(-6)
            threshold *= (1/1000000)
        #elif time[1]-time[0] >= 1:
        #    us = True
        else:
            us = False
            T_min = time[1]-time[0]
    #thresh_fft = threshold*T_min*len(time)    #convert from actual time to digital time
    #if us is True:
    #    T_min = 10**(-6)
    #    threshold *= (1/1000000)
    time = time.astype(float)
    #print(threshold)
    val = data[:, 1]
    N = len(val)
    fs = 1/np.average(np.abs(find_diff(time)))
    f_max = fs/2
    if N & 0x1:
        f_max *= (N-1)/N
    #val_bin = val>=threshold
    f_min = 2/(T_min * N**2)
    freqs = np.append(0, np.linspace(f_min, f_max, np.int(np.round(N/2))))
    mags = np.abs(np.fft.rfft(val))
    #plt.semilogx(freqs, mags)
    #plt.plot(freqs[1:], mags[1:])
    #w = 2*np.pi*freqs[1+np.argmax(mags[1:]*(freqs[1:len(mags)]<threshold))] # angular velocity of the wheel
    f_w = freqs[np.nonzero((freqs<(threshold)))][1:]
    m_w = mags[np.nonzero((freqs<(threshold)))][1:]
    w = f_w[np.argmax(m_w)]*2*np.pi
    #w = np.argmax((freqs<(threshold))*mags)[1:])
    #pwm = freqs[1+np.argmax(mags[1:]*(freqs[1:]>threshold))]
    cutoff = 3*np.round(w)*2/1000000
    b, a = signal.butter(1, cutoff, btype="low")
    res = val - signal.lfilter(b, a, val)*2/3   # level-shift proprtional to magnitude
    res = ((res*(res>1))>1)*np.ones(len(res)) # convert res to binary
    res = find_diff(res)  # find edges, preserving edge direction
    if plot is True:
        plt.plot(time[:50000], res[:50000])
        plt.plot(time[:50000], val[:50000])
    time=time[:-1]*res          # keep only the times that have an edge
    args = np.nonzero(time)
    if len(args[0]) != 0:
        time=time[args] # get rid of the elements that are 0
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
    length = min([len(rising), len(falling)])
    rising = rising[:length]
    falling = falling[:length]
    if time[0]>0:
        #print(rising[0], falling[0])
        t_on = np.abs(falling - rising)
        t_off = np.abs(falling[1:]-rising[:-1])
    else:
        #print("f")
        t_on = np.abs(falling[1:]-rising[:-1])
        t_off = np.abs(falling-rising)
    t_on = t_on*(t_on<(1/threshold))
    t_on = t_on[np.nonzero(t_on)]
    t_off = t_off*(t_off<(1/threshold))
    t_off = t_off[np.nonzero(t_off)]
    if us is True:
        w*=1000000
    return periods, t_on, t_off, w

#@jit
def stats(values):
    """does that statistical analysis. returns the min, 1st quartile,
    mean, 2nd quartile, max, and standard error.\n
    accepts a 1-d array, and filters out outliers"""
    h = np.histogram(values, "fd")
    pos = np.nonzero(h[0]>(h[0].max()/20))[0]
    out=np.zeros(len(values))
    for i in pos:
        out+=values*((values >=h[1][i]) & (values <= h[1][i+1]))
    out = out[np.nonzero(out)]
    return np.append(np.percentile(out, [0, 25, 50, 75, 100]), np.std(out)/np.sqrt(len(out)))


def dec_plot(data, start=0, end=None, p=500000, use_times=False):
    """decimates and plots the data between start and end, limited to p data points"""
    if np.shape(data)[0]!=2:
        if np.shape(data)[-1] == 2:
            data = np.transpose(data)
        else:
            return "data has wrong shape:\t" + str(np.shape(data))
    times, values = data
    if end is None:
        length = len(values[start:])
    else:
        #values = values[start, end]
        length = end-start
    pos = np.linspace(start, start+length, length+1, dtype=int)
    d = 1
    while len(pos)/d > p:
        d+=1
    trim = len(pos)%d
    if trim != 0:
        pos = pos[:-trim]
    print(d, trim, len(pos))
    pos = pos.reshape((int(len(pos)/d), d))[:, 0]
    #values = values.reshape((int(len(values/d), )))
    if use_times is True:
        pos = times[pos]
    plt.plot(pos, values[pos])

#@jit
def moving_level_shift(data, w_length=100000, threshold=10000, plot=False, us=True):
    """does the level_shift on a limited window that moves"""
    shape = np.shape(data)
    if (shape[1]==2):
        data = np.transpose(data)
    f_min = (us*10**6) * 1/(w_length*(data[1][1]-data[1][0]))
    check = input("Minimum frequency is %d, continue (y for yes)?" % f_min)
    if (check[0] == "y") or (check[0] == "Y"):
        pass
    else:
        return("aborted")
    w_out = np.zeros(len(data[1])-w_length)
    t_out = np.zeros(len(w_out))
    #c = 0
    #while c < len(w_out):
    counts = np.arange(0, len(w_out))
    for c in counts:
        w_out[c] = level_shift(data[c:c+w_length, :], threshold=threshold, plot=False, us=us)[-1]
        t_out[c] = np.average(data[c:c+w_length, 0])
        #print(c)
        c+=1
    if plot is True:
        plt.plot(t_out, w_out)
    return t_out, w_out


def moving_rms(data, w_length=10000):
    """plots the moving rms of the data"""
    if (np.shape(data)[0] != 2) and (np.shape(data)[1]==2):
        data = np.transpose(data)
    rms_out = np.zeros(1+len(data[1])-w_length)
    t_out = data[0][:len(rms_out)]
    c = 0
    while c < len(rms_out):
        rms_out[c] += np.sqrt(np.average(data[1, c:c+w_length]**2))
        c+=1
    return t_out, rms_out

def rms(vals):
    """returns the rms of the values"""
    return np.sqrt(np.average(vals**2))

#@jit
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
    