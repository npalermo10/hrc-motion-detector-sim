### notes: delay 25ms came from "defining the computational sturcture of motion detectors, Clark et al, 2011"

import numpy as np
from sympy import Symbol
from math import ceil
import stimuli
import cv2

stim_length = 1000
brightness = 1000

stim1 = stimuli.Grating(stim_length)
stim2 = stimuli.Grating(stim_length)

stim1.add_sinusoid(frequency= 2)
signal_a = brightness*(stim1.image + 1)

stim2.add_sinusoid(frequency= 2, phase = pi/10)
signal_b = brightness*(stim2.image + 1)

# brightness = 10000
# delay = pi/2
# signal_a = brightness*np.sin(linspace(0, 4*pi, 1000)) + brightness
# signal_b = brightness*np.sin(linspace(0, 4*pi, 1000)- delay) + brightness

def rotate_image_about_point(image, angle, coords = (0,0)):
    rot_mat = cv2.getRotationMatrix2D(coords, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result    

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
        self.a_detector = Light_detector(self.a_loc)
        signal_1_rot = rotate_image_about_point(signal_1, self.ori, (self.loc[0], self.loc[1]))
        self.a_detector.get_input(signal_1_rot[0])
        
        self.b_detector = Light_detector(self.b_loc)
        signal_2_rot = rotate_image_about_point(signal_2, self.ori, (self.loc[0], self.loc[1]))
        self.b_detector.get_input(signal_2_rot[0])
        
        self.x_a_out = self.a_detector.input
        self.str_a_out = self.delay_input(self.a_detector.input)
        self.x_b_out = self.b_detector.input
        self.str_b_out = self.delay_input(self.b_detector.input)
        
        return (self.str_a_out * self.x_b_out) - (self.x_a_out * self.str_b_out)

# class Eye():
#     def __init__(self):
#         self.detectors = False

#     def add_detector()

det = R_detector(ori = 0)
det2 = R_detector(ori = 45)
det3 = R_detector(ori = 90)

plot(det.output(signal_a, signal_b))
plot(det2.output(signal_a, signal_b))
plot(det3.output(signal_a, signal_b))

