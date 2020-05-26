'''
********************************************************************
Everguard 2020
File Name: fall_detection_v1.py
Author: Kyung Lee  Version 1.0  Date: May 25 2020
Description: Fall detection using IMU data in frequency domain (PSD)
             and PCA-based subspace projection
********************************************************************
'''

import socket
import numpy as np
from scipy import signal
import collections

class IMU:

	def __init__(self, freq_val=10, udp_addr="192,168,1,114", port=44777, buf_sz=1024):
		self.freq_val = freq_val #Hz
		self.udp_addr = udp_addr
		self.port = port
		self.buf_sz = buf_sz		
		self.sampling_rate = 1.0/self.freq_val
		self.sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
		self.sock.bind((self.udp_addr, self.port))

	def get_acc(self):

		msg, addr = self.sock.recvfrom(self.buf_sz)
		msg_txt = msg.decode("utf-8")
		splt = msg_txt.split(',')

		try:
			device = splt[1]
			if device == 'IMD':

				tagID = splt[2]
				t_stamp = splt[3].split(' ')
				time = t_stamp[1]
				min_sec = time.split(':')
				min = float(min_sec[1])
				sec = float(min_sec[2])
			
				t_sec1 = min*60 + sec #millisecond
				acc1_x = float(splt[7])
				acc1_y = float(splt[8])
				acc1_z = float(splt[9])
				gyr1_x = float(splt[10])
				gyr1_y = float(splt[11])
				gyr1_z = float(splt[12])
			
				t_sec2 = round(t_sec1+self.sampling_rate,3)
				acc2_x = float(splt[15])
				acc2_y = float(splt[16])
				acc2_z = float(splt[17])
				gyr2_x = float(splt[18])
				gyr2_y = float(splt[19])
				gyr2_z = float(splt[20])
				
				return [[acc1_x,acc1_y,acc1_z],[acc2_x,acc2_y,acc2_z]]

		except IndexError:
			pass


class Fall:

	def __init__(self, eigen_dir, thresold=0.05, length=64):
		self.eigen_dir = eigen_dir
		self.threshold = threshold
		self.length = length  # length of time series signal
		self.imu = IMU()
		self.new_fq = np.linspace(0, 10, num=101, endpoint=True)		
		self.Eig = np.load(self.eigen_dir + "/eigenvec3.npy")
		self.ave = np.load(self.eigen_dir + "/mean.npy")

	def get_PSD(self, acc_sig):
		f, psd = signal.welch(acc_sig, self.imu.freq_val)
		psd_intp = np.interp(self.new_fq, f, psd)
		return psd_intp

	def isFallen(self, sig):
		Eig = self.Eig.T
		mean = self.mean.T
		sig = sig.T
		sig_ = sig - mean

		x_est, _, _, _ = np.linalg.lstsq(Eig, sig_)
		sig_est = np.matmul(Eig, x_est) + mean
		mse = (np.square(sig - sig_est)).mean(axis=None)/(np.square(sig)).mean(axis=None)

		if mse < self.threshold:
			return True
		else:
			return False

	def run(self):
		ACC_x = collection.deque(maxlen=self.length)
		ACC_y = collection.deque(maxlen=self.length)
		ACC_z = collection.deque(maxlen=self.length)

		while(True):
			acc = self.imu.get_acc
			ACC_x.append(acc[0][0],acc[1][0])
			ACC_y.append(acc[0][1],acc[1][1])
			ACC_z.append(acc[0][2],acc[1][2])

			if len(ACC_x) == self.length:
				f, psd_x = self.get_PSD(ACC_x)
				f, psd_y = self.get_PSD(ACC_y)
				f, psd_z = self.get_PSD(ACC_z)
				PSD = [(x+y+z)/3.0 for x,y,z in zip(psd_x,psd_y,psd_z)]

				if self.isFallen(PSD) == True:
					print("Falling happened!")


if __name__ == "__main__": 

	eigen_dir = "...."
	fall = Fall(eigen_dir)
	fall.run()
			














