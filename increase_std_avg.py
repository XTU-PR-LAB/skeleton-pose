# -*- coding: utf-8 -*-
from __future__ import division
import numpy


class incre_std_avg():

    def __init__(self, h_avg=0, h_std=0, n=0):
        self.avg = h_avg
        self.std = h_std
        self.n = n

    def incre_in_list(self, new_list):
        if len(new_list) == 0:
            return
        avg_new = numpy.mean(new_list)
        incre_avg = (self.n*self.avg+len(new_list)*avg_new) / \
            (self.n+len(new_list))
        std_new = numpy.std(new_list)
        incre_std = numpy.sqrt((self.n*(self.std**2+(incre_avg-self.avg)**2)+len(new_list)
                                * (std_new**2+(incre_avg-avg_new)**2))/(self.n+len(new_list)))
        self.avg = incre_avg
        self.std = incre_std
        self.n += len(new_list)

    def incre_in_value(self, value):
        incre_avg = (self.n*self.avg+value)/(self.n+1)
        incre_std = numpy.sqrt((self.n*(self.std**2+(incre_avg-self.avg)
                                        ** 2)+(incre_avg-value)**2)/(self.n+1))
        self.avg = incre_avg
        self.std = incre_std
        self.n += 1

