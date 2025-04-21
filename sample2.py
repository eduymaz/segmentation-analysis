# Ürün Kümeleme (Benzer Ürünler)
# DBSCAN kullanarak benzer sipariş geçmişine sahip ürünleri gruplandırma
# Az satılan veya alışılmadık kombinasyonlarda geçen ürünleri belirleme

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import psycopg2
from sqlalchemy import create_engine
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import base64
from io import BytesIO
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

app = FastAPI(title="Ürün Segmentasyonu API",
             description="Benzer ürünleri kümeleme ve analiz etme API'si",
             version="1.0.0")

class ProductStats(BaseModel):
    product_id: int
    product_name: str
    avg_price: float
    order_frequency: float
    avg_quantity: float
    unique_customers: int
    cluster: int

class ProductClusteringResponse(BaseModel):
    clusters: List[Dict[str, Any]]
    outliers: List[Dict[str, Any]]

def perform_product_clustering():
    # Veritabanı bağlantısı
    user = 'postgres'
    password = "1352"
    host = 'localhost'
    port = '5432'
    database = 'nortwind'

    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}") 

    # Ürün verilerini çekme
    query = """
    SELECT 
        p.product_id,
        p.product_name,
        AVG(od.unit_price) as avg_price,
        COUNT(DISTINCT o.order_id) as order_frequency,
        AVG(od.quantity) as avg_quantity,
        COUNT(DISTINCT o.customer_id) as unique_customers
    FROM 
        products p
    INNER JOIN 
        order_details od ON p.product_id = od.product_id
    INNER JOIN 
        orders o ON od.order_id = o.order_id
    GROUP BY 
        p.product_id, p.product_name
    HAVING 
        COUNT(DISTINCT o.order_id) > 0
    """

    df = pd.read_sql_query(query, engine)

    # Özellik seçimi ve ölçeklendirme
    features = ['avg_price', 'order_frequency', 'avg_quantity', 'unique_customers']
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optimal eps değerini belirleme
    def find_optimal_eps(X_scaled, min_samples=3):
        neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
        distances,_ = neighbors.kneighbors(X_scaled)
        distances = np.sort(distances[:, min_samples-1])
        kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
        return distances[kneedle.elbow]

    optimal_eps = find_optimal_eps(X_scaled)
    
    # DBSCAN kümeleme
    dbscan = DBSCAN(eps=optimal_eps, min_samples=3)
    df['cluster'] = dbscan.fit_predict(X_scaled)

    # Silhouette skorlarını hesapla
    silhouette_scores = []
    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Görselleştirme
    plt.figure(figsize=(12, 8))
    plt.scatter(df['order_frequency'], df['avg_price'], c=df['cluster'], cmap='viridis', s=60)
    plt.xlabel("Sipariş Sıklığı")
    plt.ylabel("Ortalama Fiyat")
    plt.title("Ürün Segmentasyonu (DBSCAN)")
    plt.grid(True)
    plt.colorbar(label='Küme No')
    plt.show()  # Görselleştirmeyi göster
    
    # Silhouette skorları için görselleştirme
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Silhouette Skoru')
    plt.title('Optimal Küme Sayısı Analizi')
    plt.grid(True)
    plt.show()  # Görselleştirmeyi göster

    # Sonuçları hazırlama
    results = {
        'clusters': [],
        'outliers': [],
        'visualization': None  # API yanıtında görselleştirme artık gerekli değil
    }

    # Her küme için istatistikler
    for cluster in df['cluster'].unique():
        if cluster != -1:  # Aykırı değerleri hariç tut
            cluster_data = df[df['cluster'] == cluster]
            cluster_stats = {
                'cluster_number': int(cluster),
                'product_count': int(len(cluster_data)),
                'avg_price': float(cluster_data['avg_price'].mean()),
                'avg_order_frequency': float(cluster_data['order_frequency'].mean()),
                'avg_quantity': float(cluster_data['avg_quantity'].mean()),
                'products': cluster_data[['product_id', 'product_name', 'avg_price', 'order_frequency', 'avg_quantity', 'unique_customers']].to_dict('records')
            }
            results['clusters'].append(cluster_stats)

    # Aykırı değerleri (küme -1) analiz etme
    outliers = df[df['cluster'] == -1]
    results['outliers'] = outliers[['product_id', 'product_name', 'avg_price', 'order_frequency', 'avg_quantity', 'unique_customers']].to_dict('records')

    return results

@app.get("/api/product-clustering", 
         response_model=ProductClusteringResponse,
         summary="Ürün segmentasyonu sonuçlarını getir",
         description="Benzer ürünleri analiz eder ve kümeleme sonuçlarını döndürür")
async def get_product_clustering_results():
    try:
        results = perform_product_clustering()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 