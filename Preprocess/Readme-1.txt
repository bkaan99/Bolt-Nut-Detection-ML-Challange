extracter.py 

Challange Raw Dataset\images 'dan video okuyor framleri ayırıp resim çıktısı olarak -----> extracter-output dosyasına çıkartıyor

extracter-output dosyasına Challange Raw Dataset içindeki annotation dosyasını da kopyalanması gerekiyor bu işlemi de elle yapıyoruz.

Ardından verisetimiz Roboflow da Yolov7 veri setine dönüşmek için hazır hale geliyor.

extracter-output roboflow yükledikten sonra orada augment, preprocess, test-train-valid bölümleme olaylarını yapabiliyoruz. 
Link: https://app.roboflow.com/bilge-kaan-grgen-j9nkx/obj-2/2

Bu linkten indirdiğimiz dosyalarda roboflow-output klasöründe bulunmaktadır. Yolov7 için eğitime girecek veriler ve label'ları içerisindedir.






