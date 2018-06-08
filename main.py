### notes: delay 25ms came from "defining the computational sturcture of motion detectors, Clark et al, 2011"

import numpy as np
import matplotlib.patches as mpatches

blue, green, yellow, orange, red, purple = [(0.30, 0.45, 0.69), (0.33, 0.66, 0.41), (0.83, 0.74, 0.37), (0.78, 0.50, 0.16), (0.77, 0.31, 0.32), (0.44, 0.22, 0.78)]


class Light_detector():
    ''' a simple light detector. takes in light at discrete levels '''
    def __init__(self, location = [0,0]):
        self.location = location
        
    def get_input(self, signal):
        self.input = poisson(signal) - mean(signal)
                   
class HRC():
    ''' reichardt motion detector model '''
    def __init__(self, loc = [0,0], ori = 0, dist = 50, color = blue):
        self.ori = ori ## orientation given as degrees angle (0 thru 90)
        self.loc = loc
        self.dist = dist
        self.color = color
        self.get_detector_coords()
        self.a_detector = Light_detector(self.a_loc)
        self.b_detector = Light_detector(self.b_loc)
        self.output = False
        
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
    
    def get_output(self, signal_1, signal_2):
        self.a_detector.get_input(signal_1)
        self.b_detector.get_input(signal_2(self.ori))
        self.x_a_out = self.a_detector.input
        self.str_a_out = self.delay_input(self.a_detector.input)
        self.x_b_out = self.b_detector.input
        self.str_b_out = self.delay_input(self.b_detector.input)
        self.output = (self.str_a_out * self.x_b_out) - (self.x_a_out * self.str_b_out)

class Eye():
    def __init__(self):
        self.detectors = []
        self.mot_dir = []
        self.mot_mag = []
       
    def add_detector(self, orientation, clr):
        self.detectors.append(HRC(ori= orientation, color=clr))

    def calc_motion(self,sig_a, sig_b):
        for d in self.detectors:
            d.get_output(sig_a, sig_b)
        v = np.array([[cos(d.ori)*d.output, sin(d.ori)*d.output] for d in self.detectors])
        vdotx = np.zeros([v.shape[0], v.shape[2]])
        vdoty = np.zeros([v.shape[0], v.shape[2]])
        for i_det, det in enumerate(v):
            for t in arange(det.shape[1]):
                vdotx[i_det][t] = np.dot([det[0][t], det[1][t]], [1,0])
                vdoty[i_det][t] = np.dot([det[0][t], det[1][t]], [0,1])
        v_x = mean(vdotx, axis = 0)
        v_y = mean(vdoty, axis = 0)
        self.mot_dir = arctan2(v_y, v_x)
        self.mot_mag = sqrt(v_x**2 + v_y**2)
                        
detectors = [[0, blue],
            [pi/4, green],
            [pi/2, red]]
            
            
eye = Eye()
for d_angle, d_color in detectors:
    eye.add_detector(d_angle, d_color)

brightness = 10000
contrast = 1000
delay = pi/5
signal_a = contrast*np.sin(linspace(0, 4*pi, 1000)) + brightness #sf = stretch factor
signal_b =  lambda angle: brightness*np.sin(linspace(0, 4*pi, 1000) - delay*cos(angle)) + brightness

eye.calc_motion(signal_a, signal_b)
