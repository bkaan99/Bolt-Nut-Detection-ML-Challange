# ML Object Detection and Counting Challange
## @Author : Bilge Kaan Gürgen

## Başlamadan Önce

---

- Verilen [**Github**](https://github.com/Stroma-Vision/machine-learning-challenge) reposunu incelendi.
- **640x640** pixel boyutunda **Test, Train, Val** olmak üzere **3 adet video** dosyası verildi.
- **COCO** formatında Label işlemi yapılmış.
- **Bolt ,Nut** olarak 2 Class tespiti ve bunların sayılması istenmektedir.

Örnek verilmiş Bounding Box ile Object Detection yapılmış projenin aslında bizden istenen görüntüsü

![https://github.com/Stroma-Vision/machine-learning-challenge/blob/main/sample.gif?raw=true](https://github.com/Stroma-Vision/machine-learning-challenge/blob/main/sample.gif?raw=true)

**JSON formatında verilen örnek annotation dosyası**

---

```json
{
    "info": {
        "description": "Nuts and Bolts Dataset",
        "url": "",
        "version": "1.0",
        "year": 2023,
        "contributor": "Dogukan",
        "date_created": "2023/01/18"
    },
    "licenses": [],
    "images": [
        {
            "file_name": "0000.jpg",
            "width": 640,
            "height": 640,
            "id": 1
        },
				.................
				{
            "file_name": "1799.jpg",
            "width": 640,
            "height": 640,
            "id": 1800
        }
    ]
```

# 1-Preprocess

---

Bize Verilen videolardan örneğin 60 Saniyelik 30FPS Video’nun 1800 Frame’den oluştuğu bilinmektedir. Bunun sonucunda alacağımız video dosyasına ait `JSON Annotation` dosyasında da o kadar `id` sayısında olması gerekiyor. 

Bunun için Video’yu Frame’lerine bölerek çıkartan Python Kodu yazdım. `Extractor.py`

### Output dosyalarının olması gereken görüntüsü

![Untitled](readme_media/Untitled.png)

### **Adımlar**

1. Bize verilen verisetini [buradan](https://github.com/Stroma-Vision/machine-learning-challenge/releases) indirmeliyiz.
2. Preprocess Klasöründe ki `extractor.py` ile bu verisetinde ki video dosyasını frame’lere ayırıyoruz. 
3. Ayrışmış frame klasörü içerisine bize verilen dataset [buradan](https://github.com/Stroma-Vision/machine-learning-challenge/releases) içerisinde hangi video frame’lere  ayrıştırılmış ise ona ait JSON annotation dosyası da o klasöre kopyalanmalıdır.
4. Bu aşamalardan sonra görüntüsü [buradaki](https://github.com/bkaan99/challange/blob/master/readme_media/Untitled.png) gibi olmalı.
5. Ayırılmış Frameleri [ROBOFLOW](https://app.roboflow.com/) sitesine yüklüyoruz. Yükleme aşamalarında sadece test, train, valid bölme işlemini yapmanız yeterlidir. [Adım Adım Yükleme aşamaları](https://roboflow.com/convert/coco-json-to-yolo-darknet-txt)
    
    ![Untitled](readme_media/Untitled%201.png)
    
6. Yükledikten sonra Roboflow üzerinde verisetini yolov7 formatında Export etmeliyiz. 
7. Export aşamasında bilgisayarıma indirmeden download code ile Colab içerisine dahil edebiliriz.

Oluşan verisetine bu [Link](https://app.roboflow.com/bilge-kaan-grgen-j9nkx/obj-2/2)’ den ulaşabilirsiniz. 

# 2- Train

---

`Eğitim Yolov7 kullanılarak yapılacaktır.` [Yolov7](https://github.com/wongkinyiu/yolov7)

Training aşamasını `Training` klasörü içerisinde ki notebook dosyasından yapıyoruz. Bu aşamayı Google Colab üzerinde yapacağız.

### **Adımlar**

1. Colab üzerinde belirtilen notebook dosyasını açın. [Colab Training Notebook](https://colab.research.google.com/drive/1N1x6fRbfz6tga9kPmcIBTE8yS0EQNZ4x#scrollTo=6AGhNOSSHY4_)
2. Colab içerisinde belirtilen adımları hücreleri çalıştırarak ilerleyin.
3. Hiçbir değişiklik yapmadan (batch size, epoch sayısı) çalıştırarak eğitebilirsiniz. Overfit (Aşırı Öğrenme) durumunda `yolov7.yaml` dosyasında ki learning rate değerini değiştirerek ilerleyebilirsiniz.
4. Eğitim sonucunda model dosyalarınız Colab içerisindeki `runs/train/weights` ‘de kaydedilmektedir. Biz en iyi sonucu olan model dosyasını yani best.pt dosyasını kullanacağız..
5. `best.pt` dosyanızı Drive içerisine taşımayı unutmayın. Çünkü bu dosyayı predict aşamasında kullanmamız gerekmektedir. 

### Hücre Açıklamaları

---

```python
!nvidia-smi
```

**`nvidia-smi`** (NVIDIA System Management Interface) bir komut satırı aracıdır ve NVIDIA GPU'larının sistemle ilgili bilgilerini görüntülemek için kullanılır. Bu araç, GPU bellek kullanımını, işlemci kullanımını, sıcaklığı, saat hızını, güç tüketimini ve diğer çeşitli performans ölçümlerini görüntülemeyi sağlar.

```python
!git clone https://github.com/SkalskiP/yolov7.git
%cd yolov7
!git checkout fix/problems_associated_with_the_latest_versions_of_pytorch_and_numpy
!pip install -r requirements.txt
```

Yolov7 reposu indirilir ve requirements.txt dosyasında ki kütüphaneler yüklenir.

```python
!pip install roboflow 

from roboflow import Roboflow
rf = Roboflow(api_key="x5tKxqHm8Of46UZ2y5eV")
project = rf.workspace("bilge-kaan-grgen-j9nkx").project("obj-2")
dataset = project.version(2).download("yolov7")
```

Roboflow API'sini kullanarak belirli bir Roboflow projesindeki veri kümesini indiriyor.

```python
%cd /content/yolov7
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
```

Yolov7 modelinin eğitiminde kullanılabilecek önceden eğitilmiş bir ağırlık dosyasını indirir.

```python
%cd /content/yolov7
!python train.py --batch 16 --epochs 55 --data {dataset.location}/data.yaml --weights 'yolov7_training.pt' --device 0
```

Bu kod, YOLOv7 modeli için eğitim yapmak üzere **`train.py`** dosyasını çalıştırır.

- **`-batch 16`** parametresi, her bir eğitim adımında kullanılacak olan görüntü sayısını belirtir.
- **`-epochs 55`** parametresi, toplamda kaç epoch (bir epoch, tüm eğitim verilerinin bir kez geçtiği durumu ifade eder) eğitim yapılacağını belirtir.
- **`-data {dataset.location}/data.yaml`** parametresi, veri setinin konumunu belirten YAML dosyasının konumunu belirtir.
- **`-weights 'yolov7_training.pt'`** parametresi, önceden eğitilmiş ağırlıkların konumunu belirtir.
- **`-device 0`** parametresi, kullanılacak olan GPU numarasını belirtir.

Eğitim aşamasına ait görüntü.

```python
.............
.............
Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     53/54     12.7G   0.03008  0.004752   0.00143   0.03626        17       640: 100% 71/71 [01:03<00:00,  1.11it/s]
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100% 9/9 [00:05<00:00,  1.56it/s]
                 all         279         698       0.966       0.937       0.988       0.818

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     54/54     12.7G   0.01908  0.004578 0.0009224   0.02458        27       640: 100% 71/71 [01:03<00:00,  1.11it/s]
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100% 9/9 [00:07<00:00,  1.25it/s]
                 all         279         698       0.969       0.979       0.987       0.831
                bolt         279         570       0.984       0.965       0.992       0.863
                 nut         279         128       0.955       0.992       0.982         0.8
55 epochs completed in 1.108 hours.

Optimizer stripped from runs/train/exp/weights/last.pt, 74.8MB
Optimizer stripped from runs/train/exp/weights/best.pt, 74.8MB
```

```python
%load_ext tensorboard
%tensorboard --logdir /content/yolov7/runs/train
```

**`load_ext`**Tensorboard'u Colab'da kullanmamıza olanak sağlar. 

train klasörü altındaki log dosyalarına erişmek için **`logdir`** parametresi kullanılmıştır. Bu sayede, modelin eğitimi sırasında kaydedilen metrikleri ve görselleştirmeleri gözlemleyebiliriz.

![Untitled](readme_media/Untitled%202.png)

```python
from google.colab import drive
drive.mount('/content/drive')
```

Google drive bağlantımızı yapıyoruz. 

### TensorBoard Çıktıları & mAP, Precision, Recall

---

Tensorboard'daki "map" değeri, modelin ortalama hassasiyet değerini (mAP) ifade eder. Bu değer, modelin sınıflandırma doğruluğunu ölçmek için kullanılır. mAP, tüm sınıfların hassasiyet değerlerinin ortalamasıdır.

"map05:0.95" değeri, 0.5 ile 0.95 arasındaki tüm eşik değerleri için ortalama hassasiyet değerini ifade eder. Bu değer, modelin daha geniş bir aralıkta nasıl performans gösterdiğini gösterir.

Precision ve recall değerleri, modelin doğruluğunu daha ayrıntılı bir şekilde analiz etmek için kullanılan diğer ölçümlerdir. Precision, modelin doğru sınıflandırılan nesnelerin sayısının, toplam sınıflandırılan nesnelerin sayısına oranını ifade eder. Recall ise, modelin doğru sınıflandırılan nesnelerin sayısının, toplam gerçek nesne sayısına oranını ifade eder. Bu ölçümler, modelin hangi sınıfların daha iyi sınıflandırıldığını ve hangi sınıfların daha kötü performans gösterdiğini belirlemeye yardımcı olur.

### **Model Çıktı dosyası**

> `best.pt` file [drive link](https://drive.google.com/file/d/1vjdGoTi0UGOjXEuRv3W6EqY0UJVcbay6/view?usp=sharing)
> 

### Runtime Hatası için çözüm

Train komutunu çalıştırırken bazen `RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)` belirtilen runtime hatası alabiliyoruz. Bunun için çok güzel ve kolay bir çözüm yolu stackoverflow’da anlatılmış.  [Çözüm](https://stackoverflow.com/questions/74372636/indices-should-be-either-on-cpu-or-on-the-same-device-as-the-indexed-tensor)

# 3-Prediction

---

Model çıktılarımız aldıktan sonra objelerin saydırılması, Bounding Box’lar içerisinde doğruluk değerleri ve hangi obje olduğunu göstereceğiz.

1. [Colab Prediction Notebook](https://colab.research.google.com/drive/1lW7p1Tw6SE8iW0mEsd42w5z_AfVp-ESg?usp=sharing) ‘ linkinden Colab Notebook’una ulaşıyoruz.
2. Colab içerisinde ki hücrelere ait adımları takip ediyoruz.
3. Adımlardan drive bağlama adımından sonra verilmiş drive linki üzerinden [Drive](https://drive.google.com/file/d/1vWEsg5wmFAy4PqPwugHecPzBAVgnpn0B/view?usp=sharing) yada [Prediction/**ObjectTracker**/](https://github.com/bkaan99/challange/tree/master/Prediction/ObjectTracker)  Dosya yolu içerisinde ki `pred.py` , `sort.py` , `requirements.txt` ve `requirements_gpu.txt` 4 adet dosyayı Colab içerisinde oluşan yolov7 klasörü içerisine atıyoruz.
4. Takip ettikten sonra linkler üzerinde’ki dosyaları kendi Colab dosya yollarınıza indirerek prediction işlemlerinizi yapabilirsiniz.

# 4 - Sonuç

---

Colab ile Test notebook dosyası sonrası projede bizden istenen çıktısı verilmiştir.

[Final Result](https://drive.google.com/file/d/1BtbForG_GCb5M8spGb8GJssTC9nlwIwJ/view?usp=sharing)

## Detaylı Rapor

---

[Detaylı Rapor](https://www.notion.so/Milestone-ML-Challenge-41ac39fbf00e4cfbb13ed2021c5b0e07)

# Teşekkür

---

Bildiğimiz bilgileri pekiştirerek araştırma seviyemizi daha da artırdığı için bu challange adına çok memnunum. Herkese iyi çalışmalar 🙂