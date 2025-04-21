# Segmentasyon Analizi Projesi

Bu proje, Northwind veritabanı üzerinde farklı segmentasyon analizleri yaparak, müşteri, ürün, tedarikçi ve ülke bazlı gruplandırmaları incelemektedir. DBSCAN algoritması kullanılarak yapılan kümeleme analizleri, sıra dışı davranışları ve önemli segmentleri ortaya çıkarmaktadır.

## Proje Katkı Sağlayanları

- Elif Barutçu
- Elif Erdal
- Didar Arslan
- Elif Duymaz Yılmaz
- Deniz Tunç
- Hatice Nur Eriş
- Elif Özbay

## Proje Yapısı

Proje, aşağıdaki ana bileşenlerden oluşmaktadır:

1. **sample2.py**: Ürün Segmentasyonu
   - Ürünlerin satış performansına göre gruplandırılması
   - Az satılan veya alışılmadık kombinasyonlarda geçen ürünlerin tespiti

2. **sample3.py**: Tedarikçi Segmentasyonu
   - Tedarikçilerin sağladıkları ürünlerin satış performansına göre gruplandırılması
   - Az katkı sağlayan veya sıra dışı tedarikçilerin tespiti

3. **sample4.py**: Ülke Segmentasyonu
   - Farklı ülkelerden gelen siparişlerin gruplandırılması
   - Sıra dışı sipariş alışkanlığı olan ülkelerin tespiti

4. **api.py**: Birleştirilmiş API
   - Tüm segmentasyon analizlerini tek bir API üzerinden sunma
   - Swagger UI ile kolay kullanım

## Teknolojiler

- Python
- FastAPI
- PostgreSQL
- scikit-learn (DBSCAN, KMeans)
- pandas
- matplotlib
- SQLAlchemy

## Proje Akışı

1. **Veri Toplama**:
   - PostgreSQL veritabanından ilgili tablolar üzerinden veri çekme
   - Her segmentasyon için özel SQL sorguları kullanma

2. **Veri Ön İşleme**:
   - Eksik verilerin temizlenmesi
   - Özellik seçimi ve ölçeklendirme
   - StandardScaler ile veri normalizasyonu

3. **Kümeleme Analizi**:
   - DBSCAN algoritması ile kümeleme
   - Optimal eps değerinin KneeLocator ile belirlenmesi
   - Silhouette skorları ile küme kalitesinin analizi

4. **Görselleştirme**:
   - Kümeleme sonuçlarının 2D grafiklerle gösterimi
   - Silhouette skorları grafiği
   - Küme dağılımlarının renk kodlaması

5. **API Entegrasyonu**:
   - FastAPI ile RESTful API oluşturma
   - Swagger UI ile dokümantasyon
   - JSON formatında sonuç döndürme

## Kullanım

1. Ortam kurulumu yapın:
#### pip ile kurulum:
   ```bash
   # Gerekli paketleri yükle
   pip install -r requirements.txt

   # Sanal ortam oluştur (opsiyonel)
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # veya
   .\venv\Scripts\activate  # Windows
   ```

#### conda ile kurulum:
   ```bash
   # Conda ortamını oluştur
   conda env create -f environment.yml

   # Ortamı aktifleştir
   conda activate problem_env
   ```

2. Veritabanı bağlantısını kurun:
   ```python
   user = ''
   password = ""
   host = 'localhost'
   port = ''
   database = ''
   ```

3. API'yi başlatın:
   ```bash
   uvicorn api:app --reload
   ```

4. Swagger UI'a erişin:
   - Tarayıcınızda `http://localhost:8000/docs` adresine gidin
   - İstediğiniz segmentasyon analizini seçin ve çalıştırın

## Endpoint'ler

- `/api/product-clustering`: Ürün segmentasyonu
- `/api/supplier-clustering`: Tedarikçi segmentasyonu
- `/api/country-clustering`: Ülke segmentasyonu

Her endpoint, kümeleme sonuçlarını ve aykırı değerleri JSON formatında döndürür.

## Sonuçlar

Her analiz sonucunda:
- Kümeleme sonuçları
- Aykırı değerler
- Görselleştirmeler
- İstatistiksel özetler

elde edilir. 
