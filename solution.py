import os
import sys
import csv
import numpy as np
from scipy import stats
import random
import math
import matplotlib.pyplot as plt

class Data:
    def __init__(self,data_path):
        self._x = 0
        self.data_path = data_path
        self.data_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
        self.data = {}

    def load(self):
        for each in self.data_files:
            x, y = np.loadtxt(os.path.join(self.data_path,each), delimiter=',', skiprows=1,usecols=(0,1), unpack=True, dtype=float)
            self.data[each] = {}
            self.data[each]['x'] = x
            self.data[each]['y'] = y

    def getDatasetNames(self):
        return self.data.keys()
        
    def getData(self,dataset_name):
        return self.data[dataset_name]
        
    def getFeature(self):
        print ""

    def getLabel(self):
        print ""

class Learn:
    
    def __init__(self):
        self._x = 0
        self.learning_rate = 0.01
        self.reg = 1

    
    '''
    Using MLE, assuming variance = theta0+theta1*x+theta2*x**2+theta3*x**3
    '''
    def mle(self,x_array,y_array,order):

        loss_func = {
            1:(lambda x_array,y_array,a,b,reg,thetas:  sum([(y_array[i]-a*x_array[i]-b)**2/(2*(thetas[0]+thetas[1]*x_array[i])**2)-math.log(1/(math.sqrt(2*math.pi)*abs(thetas[0]+thetas[1]*x_array[i]))) for i in xrange(len(x_array))]) + reg*(a**2 + b**2 + sum([t**2 for t in thetas])) ),
            2:(lambda x_array,y_array,a,b,reg,thetas:  sum([(y_array[i]-a*x_array[i]-b)**2/(2*(thetas[0]+thetas[1]*x_array[i]+thetas[2]*x_array[i]**2)**2)-math.log(1/(math.sqrt(2*math.pi)*abs(thetas[0]+thetas[1]*x_array[i]+thetas[2]*x_array[i]**2))) for i in xrange(len(x_array))]) + reg*(a**2 + b**2 + sum([t**2 for t in thetas])) ),
            3:(lambda x_array,y_array,a,b,reg,thetas:  sum([(y_array[i]-a*x_array[i]-b)**2/(2*(thetas[0]+thetas[1]*x_array[i]+thetas[2]*x_array[i]**2+thetas[3]*x_array[i]**3)**2)-math.log(1/(math.sqrt(2*math.pi)*abs(thetas[0]+thetas[1]*x_array[i]+thetas[2]*x_array[i]**2+thetas[3]*x_array[i]**3))) for i in xrange(len(x_array))]) + reg*(a**2 + b**2 + sum([t**2 for t in thetas])) ),
        }

        partial_d = {
                    1:{
                      "a": (lambda x,y,a,b,reg,thetas: x*(a*x+b-y)/(thetas[0]+thetas[1]*x)+2*reg*a),
                      "b": (lambda x,y,a,b,reg,thetas:(a*x+b-y)/(thetas[0]+thetas[1]*x)+2*reg*b),
                      "theta0":(lambda x,y,a,b,reg,thetas:-(a*x+b-y)**2/((thetas[0]+thetas[1]*x)**3) - 1/(thetas[0]+thetas[1]*x)+2*reg*thetas[0]),
                      "theta1":(lambda x,y,a,b,reg,thetas:-x*(a*x+b-y)**2/((thetas[0]+thetas[1]*x)**3) - 1/x/(thetas[0]+thetas[1]*x+2*reg*thetas[1]))
                      },
                    2:{
                      "a": (lambda x,y,a,b,reg,thetas:x*(a*x+b-y)/(thetas[0]+thetas[1]*x+thetas[2]*(x**2))+2*reg*a),
                      "b": (lambda x,y,a,b,reg,thetas:(a*x+b-y)/(thetas[0]+thetas[1]*x+thetas[2]*(x**2))+2*reg*b),
                      "theta0": (lambda x,y,a,b,reg,thetas:-(y-a*x-b)**2/(thetas[0]+thetas[1]*x+thetas[2]*(x**2) + 1/(thetas[0]+thetas[1]*x+thetas[2]*(x**2)))+2*reg*thetas[0]),
                      "theta1": (lambda x,y,a,b,reg,thetas:-x*(y-a*x-b)**2/(thetas[0]+thetas[1]*x+thetas[2]*(x**2)+x/(thetas[0]+thetas[1]*x+thetas[2]*(x**2)))+2*reg*thetas[1]),
                      "theta2": (lambda x,y,a,b,reg,thetas:-(x**2)*((y-a*x-b)**2)/(thetas[0]+thetas[1]*x+thetas[2]*(x**2))+(x**2)/(thetas[0]+thetas[1]*x+thetas[2]*(x**2))+2*reg*thetas[2])
                      },
                    3:{
                      "a": (lambda x,y,a,b,reg,thetas:x*(a*x+b-y)/(thetas[0]+thetas[1]*x+thetas[2]*(x**2)+thetas[3]*(x**3)) + 2*reg*a),
                      "b": (lambda x,y,a,b,reg,thetas:(a*x+b-y)/(thetas[0]+thetas[1]*x+thetas[2]*(x**2)+thetas[3]*(x**3)) + 2*reg*b),
                      "theta0": (lambda x,y,a,b,reg,thetas:-(y-a*x-b)**2/(thetas[0]+thetas[1]*x+thetas[2]*(x**2) + thetas[3]*(x**3)) + 1/(thetas[0]+thetas[1]*x+thetas[2]*(x**2)+thetas[3]*(x**3)) + 2*reg*thetas[0]),
                      "theta1": (lambda x,y,a,b,reg,thetas:-(y-a*x-b)**2/(thetas[0]+thetas[1]*x+thetas[2]*(x**2)+thetas[3]*(x**3))+x/(thetas[0]+thetas[1]*x+thetas[2]*(x**2)+thetas[3]*(x**3)) + 2*reg*thetas[1]),
                      "theta2": (lambda x,y,a,b,reg,thetas:-(x**2)*(y-a*x-b)**2/(thetas[0]+thetas[1]*x+thetas[2]*(x**2)+thetas[3]*(x**3))+(x**2)/(thetas[0]+thetas[1]*x+thetas[2]*(x**2)+thetas[3]*(x**3))+2*reg*thetas[2]),
                      "theta3": (lambda x,y,a,b,reg,thetas:-(x**3)*(y-a*x-b)**2/(thetas[0]+thetas[1]*x+thetas[2]*(x**2)+thetas[3]*(x**3))+(x**3)/(thetas[0]+thetas[1]*x+thetas[2]*(x**2)+thetas[3]*(x**3))+2*reg*thetas[3])
                      },
          }
        a,b,optimal_a,optimal_b,theta0 = random.random(),random.random(),0,0,random.random()
        thetas = [theta0]
        
        if order>=1: theta1 = random.random();opt_theta1 = 0;thetas.append(theta1)
        if order>=2: theta2 = random.random();opt_theta2 = 0;thetas.append(theta2)
        if order>=3: theta3 = random.random();opt_theta3 = 0;thetas.append(theta3)
        loss = 2**31

        for i in xrange(len(x_array)):
            x,y = x_array[i],y_array[i]
            a = a - self.learning_rate * partial_d[order]['a'](x,y,a,b,self.reg,thetas)
            b = b - self.learning_rate * partial_d[order]['b'](x,y,a,b,self.reg,thetas)
            theta0 = theta0 - self.learning_rate * partial_d[order]['theta0'](x,y,a,b,self.reg,thetas)
            if order>=1:
                theta1 = theta1 - self.learning_rate * partial_d[order]['theta1'](x,y,a,b,self.reg,thetas)
            if order>=2:
                theta2 = theta2 - self.learning_rate * partial_d[order]['theta2'](x,y,a,b,self.reg,thetas)
            if order>=3:
                theta3 = theta3 - self.learning_rate * partial_d[order]['theta3'](x,y,a,b,self.reg,thetas)

            curr_loss = loss_func[order](x_array,y_array,a,b,self.reg,thetas)

            if curr_loss<loss:
                self.learning_rate*=1.05
                if order>=1:
                    optimal_a,optimal_b,opt_theta0,opt_theta1 = a,b,theta0,theta1
                if order>=2:
                    optimal_a,optimal_b,opt_theta0,opt_theta1,opt_theta2 = a,b,theta0,theta1,theta2
                if order>=3:
                    optimal_a,optimal_b,opt_theta0,opt_theta1,opt_theta2,opt_theta3 = a,b,theta0,theta1,theta2,theta3

            else:
                self.learning_rate*=0.5
                if order>=1:
                    a,b,theta0,theta1 = optimal_a,optimal_b,opt_theta0,opt_theta1 
                if order>=2:
                    a,b,theta0,theta1,theta2 = optimal_a,optimal_b,opt_theta0,opt_theta1,opt_theta2 
                if order>=3:
                    a,b,theta0,theta1,theta2,theta3 = optimal_a,optimal_b,opt_theta0,opt_theta1,opt_theta2,opt_theta3 
            
            loss = curr_loss
            print curr_loss
        print "Output:"
        #print a,b,theta,self.learning_rate
        print optimal_a,optimal_b,opt_theta0,opt_theta1,opt_theta2
        
        # Evaluation
        # Perform Shapiro-Wilk test
        # Which tests the null hypothesis that the data was drawn from a normal distribution.
        normalized_array = np.array([(y_array[i]-optimal_a*x_array[i]-optimal_b)/(opt_theta2*(x_array[i]**2)+opt_theta1*x_array[i]+opt_theta0) for i in range(len(x_array))])

        print stats.shapiro(normalized_array)
        plt.plot(normalized_array,[1]*len(normalized_array),'ro')
        #plt.show()
        #plt.plot(list(x_array),list(y_array),'ro')
        plt.show()

    def mleLossNonlinear(self,x_array,y_array,a,b,theta0,theta1,theta2):
        loss = 0
        for i in range(len(x_array)):
            x = x_array[i]
            y = y_array[i]
            loss+=(y-a*x-b)**2/(2*(theta0*(x**2)+theta1*x+theta2)**2) + math.log(abs(theta0*(x**2)+theta1*x+theta2)) + self.reg_cof*(a**2+b**2+theta0**2+theta1**2+theta2**2)
        return loss



    '''
    Using MLE, assuming variance = theta0*x**2+theta1*x+theta2
    '''
    def mleWithSgdNonlinear(self,x_array,y_array):
        a,b,theta0,theta1,theta2,loss = random.random(),random.random(),random.random(),random.random(),random.random(),2**31
        optimal_a,optimal_b,optimal_t0,optimal_t1,optimal_t2 = 0,0,0,0,0
        for i in xrange(len(x_array)):
            x,y = x_array[i],y_array[i]
            a = a - self.learning_rate * (x*(a*x+b-y)/(theta0*x**2+theta1*x+theta2) + self.reg_cof*a)
            b = b - self.learning_rate * ((a*x+b-y)/(theta0*x**2+theta1*x+theta2) + self.reg_cof*b)
            theta0 = theta0 - self.learning_rate * ((- (x**2) * ((y-a*x-b)**2)/((theta0*(x**2)+theta1*x+theta2)**3) + x**2/(theta0*(x**2)+theta1*x+theta2) ) + self.reg_cof*theta0)
            theta1 = theta1 - self.learning_rate * ((-x * ((y-a*x-b)**2)/((theta0*(x**2)+theta1*x+theta2)**3) + x/(theta0*(x**2)+theta1*x+theta2) ) + self.reg_cof*theta1) 
            theta2 = theta2 - self.learning_rate * (-((y-a*x-b)**2)/((theta0*(x**2)+theta1*x+theta2)**3) + 1/(theta0*(x**2)+theta1*x+theta2) + self.reg_cof*theta2)
            curr_loss = self.mleLossNonlinear(x_array,y_array,a,b,theta0,theta1,theta2)
            
            if curr_loss<loss:
                self.learning_rate*=1.05
                optimal_a,optimal_b,optimal_t0,optimal_t1,optimal_t2 = a,b,theta0,theta1,theta2
            else:
                self.learning_rate*=0.5
                a,b,theta0,theta1,theta2 = optimal_a,optimal_b,optimal_t0,optimal_t1,optimal_t2
            
            loss = curr_loss
            print curr_loss
        print "Output:"
        #print a,b,theta,self.learning_rate
        print optimal_a,optimal_b,optimal_t0,optimal_t1,optimal_t2
        
        # Evaluation
        # Perform Shapiro-Wilk test
        # Which tests the null hypothesis that the data was drawn from a normal distribution.
        normalized_array = np.array([(y_array[i]-optimal_a*x_array[i]-optimal_b)/(optimal_t0*(x_array[i]**2)+optimal_t1*x_array[i]+optimal_t2) for i in range(len(x_array))])

        print stats.shapiro(normalized_array)
        plt.plot(normalized_array,[1]*len(normalized_array),'ro')
        plt.show()
        #plt.plot(list(x_array),list(y_array),'ro')
        #plt.show()

    def mleLossNonlinear(self,x_array,y_array,a,b,theta0,theta1,theta2):
        loss = 0
        for i in range(len(x_array)):
            x = x_array[i]
            y = y_array[i]
            loss+=(y-a*x-b)**2/(2*(theta0*(x**2)+theta1*x+theta2)**2) + math.log(abs(theta0*(x**2)+theta1*x+theta2)) + self.reg_cof*(a**2+b**2+theta0**2+theta1**2+theta2**2)
        return loss


    '''
    Using MLE method, assuming variance=theta*x
    '''
    def mleWithSgd(self,x_array,y_array):
        a,b,theta,loss = random.random(),random.random(),random.random(),2**31
        optimal_a,optimal_b,optimal_theta = 0,0,0
        for i in xrange(len(x_array)):
            x,y = x_array[i],y_array[i]
            a = a - self.learning_rate * (1/(theta*x)*(a*x+b-y))
            b = b - self.learning_rate * (1/(theta*(x**2))*(a*x+b-y))
            theta = theta - self.learning_rate * (-((y-a*x-b)**2)/((x**2)*(theta**3)) - theta)
            curr_loss = self.mleLossFunc(x_array,y_array,a,b,theta)
            if curr_loss<=loss:
                self.learning_rate*=1.05
                optimal_a,optimal_b,optimal_theta = a,b,theta
            else:
                self.learning_rate*=0.5
                a,b,theta = optimal_a,optimal_b,optimal_theta
            loss = curr_loss

            print curr_loss

        print "Output:"
        #print a,b,theta,self.learning_rate
        print optimal_a,optimal_b,optimal_theta,len(x_array)
        
        # Evaluation
        # Perform Shapiro-Wilk test
        # Which tests the null hypothesis that the data was drawn from a normal distribution.
        normalized_array = np.array([(y_array[i]-optimal_a*x_array[i]-optimal_b)/(optimal_theta*x_array[i]) for i in range(len(x_array))])
        print stats.shapiro(normalized_array)
        #plt.plot(list(x_array),list(y_array),'ro')
        #plt.show()
        return optimal_a,optimal_b,optimal_theta

    def mleLossFunc(self,x_array,y_array,a,b,theta):
        loss = 0
        for i in range(len(x_array)):
            x = x_array[i]
            y = y_array[i]
            loss+=(y-a*x-b)**2/(2*(theta**2)*(x**2)) - math.log(1/(math.sqrt(2)*abs(theta)*abs(x)))
        return loss
        
    
    '''
    Using MLE methods, assuming variance = theta_0*x**2+theta_1*x+theta_2
    '''
    def theta(self,x,y):
        return 

    def getParams(self):
        print ""

class Evaluation:
    
    def __init__(self):
        self._x = 0

    def test(self,x_array,y_array,typeOfTest,fMean,fVar):
        # t-test
        if typeOfTest=="t_test":
            print ""

if __name__=="__main__":
    d = Data("/Users/cheny/Documents/workspace/Problem1")
    d.load()
    dataset_names = d.getDatasetNames()
    data = d.getData(dataset_names[0])
    l = Learn()
    l.mle(data['x'],data['y'],2)
