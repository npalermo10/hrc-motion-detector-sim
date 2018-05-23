### notes: delay 25ms came from "defining the computational sturcture of motion detectors, Clark et al, 2011"

import numpy as np
from sympy import Symbol
from math import ceil
import stimuli
import cv2


class Light_detector():
    ''' a simple light detector. takes in light at discrete levels '''
    def __init__(self, location = [0,0]):
        self.location = location
        
    def get_input(self, signal):
        self.input = poisson(signal) - mean(signal)

            
class R_detector():
    ''' reichardt motion detector model '''
    def __init__(self, loc = [0,0], ori = 0, dist = 50):
        self.ori = ori ## orientation given as degrees angle (0 thru 90)
        self.loc = loc
        self.dist = dist
        self.get_detector_coords()

        self.a_detector = Light_detector(self.a_loc)
        self.b_detector = Light_detector(self.b_loc)

    def delay_input(self, in_sig, b = 0.53, c = 38.9):
        t = np.array(arange(600))
        a = 1
        delay_kernel = a*(t**b)*e**(-t/c) ## from harris et al. 1999
        to_divide = cumsum(delay_kernel)[-1]
        delay_kernel = (a/to_divide)*(t**b)*e**(-t/c) ## from harris et al. 1999
        delay_sig = np.append(convolve(in_sig, delay_kernel, mode = "full"), 0 )
        to_trim = int(len(delay_sig) - len(in_sig))
        return delay_sig[:-to_trim]
        
    def get_detector_coords(self):
        self.a_loc = self.loc
        self.b_loc = [self.dist* cos(self.ori), self.dist* sin(self.ori)]
    
    def output(self, signal_1, signal_2):
        self.a_detector.get_input(signal_1)

        self.b_detector.get_input(signal_2(self.ori))
        
        self.x_a_out = self.a_detector.input
        self.str_a_out = self.delay_input(self.a_detector.input)
        self.x_b_out = self.b_detector.input
        self.str_b_out = self.delay_input(self.b_detector.input)
        
        return (self.str_a_out * self.x_b_out) - (self.x_a_out * self.str_b_out)

# class Eye():
#     def __init__(self):
#         self.detectors = False

#     def add_detector()

brightness = 10000
delay = pi/2
signal_a = brightness*np.sin(linspace(0, 4*pi, 1000)) + brightness #sf = stretch factor
signal_b =  lambda angle: brightness*np.sin(linspace(0, 4*pi, 1000) - delay*cos(angle)) + brightness

det = R_detector(ori = 0)
det2 = R_detector(ori = pi/4)
det3 = R_detector(ori = pi/2)

brightness = 10000
delay = pi/2
signal_a = brightness*np.sin(linspace(0, 4*pi, 1000)) + brightness #sf = stretch factor
signal_b =  lambda angle: brightness*np.sin(linspace(0, 4*pi, 1000) - delay*cos(angle)) + brightness

plot(det.output(signal_a, signal_b))
plot(det2.output(signal_a, signal_b))
plot(det3.output(signal_a, signal_b))

