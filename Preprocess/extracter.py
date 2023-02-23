#Extract frames from video 

import cv2 
import os

def extract_frames(video_path, output_path): # Frameleri çıkartma fonksiyonu
    vidcap = cv2.VideoCapture(video_path) # Videoyu açma
    success,image = vidcap.read() # Videoyu okuma
    count = 0000 # Başlangıç değeri
    while success: # Video bitene kadar döngü
        cv2.imwrite(os.path.join(output_path, "%04d.jpg" % count), image)   #Döngü içinde frameleri kaydetme. 
        success,image = vidcap.read() # Okuma
        print('Read a new frame: ', success) # Okuma başarılıysa ekrana yazdırma
        count += 1 # Sayacı arttırma

if __name__ == '__main__': # Main fonksiyonu
    video_path = r'C:\Users\bilge\Desktop\challenge\images\test\test.mp4' # Video yolunu belirleme
    output_path = r'C:\Users\bilge\Desktop\challenge\output' # Çıktı yolunu belirleme
    extract_frames(video_path, output_path) # Çıktıyı al

