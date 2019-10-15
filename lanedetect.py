##G201592007 김도훈
##개인 CV 공부 목적

import numpy as np
import cv2
import random
"""
영상으로부터 프레임을 받아와서 현재 차량이 속한 차선을 검출해주는 클래스
차선 검출은 다음 단계를 따름
0. 흑백영상으로 전환
1. ROI 영역을 추출 - 계산량 감소 목적
2. 가우시안 블러 적용 - 노이즈 제거 (blur_frame)
3. Canny Edge 적용 frame 생성 (canny_frame)
4. canny_frame 에서 ROI 영역 재설정
5. 허프 변환을 이용하여 선들을 검출
6. 5에서 얻은 선들에서 후보차선을 검출
	각도에 따라 수평,수직 각도의 제한 각도를 둬 필요없는 선들을 버림
7. 남은 선들을 화면에 그려줌
"""

class LaneDetector :
	def __init__(self) :
		self.fit_result = []
		self.left_fit_result = []
		self.right_fit_result = []
		self.left_lane = []
		self.right_lane = []

	def interpolation(self,lines) : 
		#print('lines shape : ',lines.shape)
		ipl = lines.reshape(lines.shape[0]*2,2)

		for line in lines :
			x1, y1, x2, y2 = line
			print('y2 = ',y2, ' y1 = ',y1)
			if np.abs(y2-y1) > 5 :
				temp = np.abs(y2-y1)
				slp = (y2-y1)/(x2-x1)
				for i in range(0,temp,5) :
					if slp > 0 :
						new_p = np.array([[int(x1 + i *slp ),int(y1+i)]])
						ipl = np.concatenate((ipl,new_p),axis=0)
					elif slp < 0 :
						new_p = np.array([[int(x1-i*slp),int(y1-i)]])
						ipl = np.concatenate((ipl,new_p),axis=0)

		return ipl

	def img2Binary(self,frame,flag = 'video') :
		if flag == 'video' : 
			return  cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) ## 테스트 영상은 RGB. 캠으로 하면 BGR로 해야하는데 flag 설정을 해야할듯
		if flag == 'camera' :
			return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	def getCannyEdges(self, frame, th1, th2) : 
		return cv2.Canny(frame, th1, th2) ## th1 = low threshold, th2 = high threshold

	def gaussian(self, frame, kernel_size = 3) : ## 가우시안 커널 적용. 기본 필터 사이즈 3x3
		return cv2.GaussianBlur(frame,(kernel_size,kernel_size),0)

	def roiSet(self, frame, vertices) :
		mask_frame = np.zeros_like(frame) ## 현재는 완전 검정색 이미지로 설정
		#print('mask frame shape ',mask_frame.shape)
		if len(frame.shape) > 2 : ## RGB 처럼 3채널 이상을 가진 프레임일 경우
			channels = frame.shape[2]
			mask_color = (255,) * channels
		else : ## 바이너리 이미지인 경우
			mask_color = 255

		cv2.fillPoly(mask_frame, vertices, mask_color) # 위에서 구한 mask_color를 vertices 영역에 뿌림 => ROI 부분에 뿌림

		roi_frame = cv2.bitwise_and(frame, mask_frame) ## ROI 영역은 255이므로 frame 상의 그 부분만 남기고 전부 0이되고 해당 결과를 roi_frame에 저장
		#print('roi_frame shape : ',roi_frame.shape)
		return roi_frame

	def getHoughLines(self, frame, ## 캐니 변환이 적용된 이미지
					  rho, theta,  ## houghtransform r(0-1) = xcos(theta) + ycos(theta) , theta = 0-180
					  thres, min_len, max_gap) :  ## 검출 threshold 값과 검출될 선의 최소,최대 폭
		hough_lines = cv2.HoughLinesP(frame, rho, theta, thres,np.array([]), min_len, max_gap)
		#hough_lines = cv2.HoughLinesP(frame, rho, theta, thres, min_len, max_gap) ## 뭔가 이상함

		#line_frame = np.zeros((frame.shape[0],frame.shape[1],3),dtype=np.uint8)
		#self.drawLines(line_frame,hough_lines)

		return hough_lines

	def updateFrame(self, frame1, frame2, ap = 0.8, bt = 1, ld = 0.0) :
		return cv2.addWeighted(frame2, ap, frame1, bt, ld)

	def drawLines(self, frame, lines, color = [255,0,0], thickness = 2) :
		for line in lines :
			for x1,y1,x2,y2 in line :
				cv2.line(frame, (x1,y1), (x2,y2), color, thickness)

	def smoothing(self, lines, p_frame) :
		lines = np.squeeze(lines)
		avrg_line = np.array([0,0,0,0])

		for idx,line in enumerate(reversed(lines)) :
			if idx == p_frame :
				break
			avrg_line += line
		avrg_line = avrg_line / p_frame

		return avrg_line

	def drawFitLine(self, frame, l_res, r_res, color = (255,0,0), thickness = 10) :
		lane = np.zeros_like(frame)

		cv2.line(lane,(int(l_res[0]),int(l_res[1])),(int(l_res[2]),int(l_res[3])),color,thickness)
		cv2.line(lane,(int(r_res[0]),int(r_res[1])),(int(r_res[2]),int(r_res[3])),color,thickness)

		new_frame = self.updateFrame(lane,frame,1,0.5)
		return new_frame


	def getRandomSample(self, lines) :
		s1 = random.choice(lines)
		s2 = random.choice(lines)

		if s2[0] == s1[0] :
			while s2[0] == s1[0] :
				s2 = random.choice(lines)

		s1 = s1.reshape(1,2)
		s2 = s2.reshape(1,2)

		s3 = np.concatenate((s1,s2),axis=1)
		s3 = s3.squeeze()

		return s3

	def getFitLine(self, frame, lines) :
		r,c = frame.shape[:2]
		default_line = cv2.fitLine(lines,cv2.DIST_L2,0,0.01,0.01)
		tx0,ty0,tx1,ty1 = default_line
		#print(' ', tx0,' ', ty0,' ', tx1,' ',ty1)  

		x0 = int(((frame.shape[0]-1)-ty1)/tx0*ty0 + tx1)
		y0 = frame.shape[0]-1 
		x1 = int(((frame.shape[0]/2+100)-ty1)/ty0*tx0 + tx1)
		y1 = int(frame.shape[0]/2+100)

		return [x0,y0,x1,y1]

	def eraseLine(self,prm, lines) :
		dist = self.getDistance(prm,lines)
		erased_lines = lines[dist<13,:]
		return erased_lines

	def getDistance(self,prm, p) :
		return np.abs(prm[0]*p[:,0] + prm[1]*p[:,1]+prm[2])/np.sqrt(prm[0]**2+prm[1]**2)

	def computeParams(self,lines) :
		x1,y1,x2,y2 = lines
		#print('x1 : ',x1,' y1 : ',y1, ' x2 : ',x2,' y2 : ',y2)

		## y = ax + b 

		a = (y2-y1)/(x2-x1)
		b = y1-a*x1
		##px + qy + r = 0

		p = a
		q = -1
		r = b
		prm = np.array([p,q,r])

		return prm

	def verification(self,prm,lines) : ## 샘플링 된 데이터로 추정한 파라미터 값과 실제 데이터 거리
		dist = self.getDistance(prm,lines) ##거리계산
		total_dist = dist.sum(axis=0) ## 전체 거리
		avrg_dist = total_dist/len(lines) ## 
		return avrg_dist

	def bestFitLine(self, frame, lines, minimum = 100) :
		ideal_line = np.array([0,0,0])

		if len(lines) != 0 :
			for i in range(30) :
				sp = self.getRandomSample(lines)
				prm = self.computeParams(sp)
				cost = self.verification(prm,lines)

				if cost < minimum :
					minimum = cost
					ideal_line = prm
				if minimum < 3 :
					break

			line_erased = self.eraseLine(ideal_line,lines)
			self.fit_result = self.getFitLine(frame,line_erased)

		else :
			if (self.fit_result[3] - self.fit_result[1])/(self.fit_result[2] - self.fit_result[0]) < 0 :
				self.left_fit_result = self.fit_result
				return self.left_fit_result
			else :
				self.right_fit_result = self.fit_result
				return self.right_fit_result

		if (self.fit_result[3] - self.fit_result[1])/(self.fit_result[2] - self.fit_result[0]) < 0 :
			self.left_fit_result = self.fit_result
			return self.left_fit_result
		else :
			self.right_fit_result = self.fit_result
			return self.right_fit_result

	def detect(self, frame) :
		ht, wt = frame.shape[:2]

		## ROI 크기 설정
		vertices = np.array([[(50,ht),(wt/2-45, ht/2+60), (wt/2+45, ht/2+60), (wt-50,ht)]], dtype=np.int32)
		roi_frame = self.roiSet(frame, vertices)

		#gray_frame = self.img2Binary(roi_frame)
		blur_frame = self.gaussian(roi_frame,3)
		canny_frame = self.getCannyEdges(blur_frame, 70, 210)

		vertices1 = np.array([[(52,ht),(wt/2-43, ht/2+62), (wt/2+43, ht/2+62), (wt-52,ht)]], dtype=np.int32)
		canny_frame = self.roiSet(canny_frame, vertices1)

		hg_lines = self.getHoughLines(canny_frame, 1, 1*np.pi/180, 30, 10, 20) ## [[[x1,y1,x2,y2]....[xn1,yn1,xn2,yn2]]
		# HoughLinesP 를 이용한 결과값이기 때문에 '선분'이 나옴. 선분의 시작점. 끝점 좌표의 리스트가 나오는것임.
		#print('hg_lines shape : ',hg_lines.shape)
		
		hg_lines = np.squeeze(hg_lines) ## [[[ .... ],]] 형태라서 [[...],] 로 줄임
		gradient = (np.arctan2(hg_lines[:,1]-hg_lines[:,3], hg_lines[:,0]-hg_lines[:,2])*180)/np.pi##기울기 구함
		## 각각의 검출 선분 대상으로 화면상에서 이루는 각도를 구함 
		#print(gradient)


		## 모든 hg_lines 내의 요소들에 대해 1,3 => y1,y2 0,2 => x1,x2 길이 차를 구해서 np.arctan2(v1,v2)로 넣음
		"""   
			  /|
		     / |
		    /  v1
		   /   |
		  0--v2-   따라서 np.arctan2 의  값은 v1/v2 가 나온다. 여기에 *180/np.pi 를 하면  각 0(기울기)가 나옴
		""" 
		### 필터링, 각도에 따라 쓸모없는 선들 제거 
		## 수평으로 펼쳐진 형태 추출
		hg_lines = hg_lines[np.abs(gradient)<160]
		gradient = gradient[np.abs(gradient)<160]
		## 수직선 형태 추출
		hg_lines = hg_lines[np.abs(gradient)>95]
		gradient = gradient[np.abs(gradient)>95]
		##제거
		left_lines = hg_lines[(gradient > 0),:]
		right_lines = hg_lines[(gradient < 0),:]
		#print("left_lines : ",left_lines.shape)
		#print('left shape : ',left_lines.shape, ' right shape : ', right_lines.shape)

		#temp_frame = np.zeros((frame.shape[0],frame.shape[1],3), dtype=np.uint8) ## 3채널 매트릭스 
		
		#print("right lines : ",right_lines.shape)
		l_itp = self.interpolation(left_lines)
		r_itp = self.interpolation(right_lines)

		lfl = self.bestFitLine(frame, l_itp)
		rfl = self.bestFitLine(frame, r_itp)

		self.left_lane.append(lfl)
		self.right_lane.append(rfl)

		if len(self.left_lane) > 10 :
			lfl = self.smoothing(self.left_lane,10)
		if len(self.right_lane) > 10 :
			rfl = self.smoothing(self.right_lane,10)

		result_frame = self.drawFitLine(frame, lfl,rfl)
		return result_frame

if __name__ == '__main__':

	cam = cv2.VideoCapture('test4.mp4')

	ld = LaneDetector()
	while True :
		ret, frame = cam.read()
		if not ret :
			exit(-1)
		if frame.shape[0] !=540: # resizing for challenge video
			frame = cv2.resize(frame, None, fx=3/4, fy=3/4, interpolation=cv2.INTER_AREA) 
		f = ld.detect(frame)
		cv2.imshow('window',f)
		if cv2.waitKey(1) == ord('q') :
			break;
