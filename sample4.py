# Ülkelere Göre Satış Deseni Analizi
# DBSCAN kullanarak ülkeleri sipariş alışkanlıklarına göre gruplandırma
# Sıra dışı sipariş alışkanlığı olan ülkeleri belirleme

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import psycopg2
from sqlalchemy import create_engine
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from sklearn.metrics import silhouette_score

app = FastAPI(title="Ülke Satış Deseni Analizi API",
             description="Ülkeleri sipariş alışkanlıklarına göre kümeleme ve analiz etme API'si",
             version="1.0.0")

class CountryStats(BaseModel):
    country: str
    total_orders: int
    avg_order_value: float
    avg_products_per_order: float
    cluster: int

class CountryClusteringResponse(BaseModel):
    clusters: List[Dict[str, Any]]
    outliers: List[Dict[str, Any]]

def perform_country_clustering():
    # Veritabanı bağlantısı
    user = 'postgres'
    password = "1352"
    host = 'localhost'
    port = '5432'
    database = 'nortwind'

    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}") 

    # Ülke verilerini çekme
    query = """
    SELECT 
        c.country,
        COUNT(DISTINCT o.order_id) as total_orders,
        AVG(od.quantity * od.unit_price) as avg_order_value,
        AVG(od.quantity) as avg_products_per_order
    FROM 
        customers c
    INNER JOIN 
        orders o ON c.customer_id = o.customer_id
    INNER JOIN 
        order_details od ON o.order_id = od.order_id
    GROUP BY 
        c.country
    HAVING 
        COUNT(DISTINCT o.order_id) > 0
    """

    df = pd.read_sql_query(query, engine)

    # Özellik seçimi ve ölçeklendirme
    features = ['total_orders', 'avg_order_value', 'avg_products_per_order']
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
    plt.scatter(df['total_orders'], df['avg_order_value'], c=df['cluster'], cmap='viridis', s=60)
    plt.xlabel("Toplam Sipariş Sayısı")
    plt.ylabel("Ortalama Sipariş Tutarı")
    plt.title("Ülke Satış Deseni Segmentasyonu (DBSCAN)")
    plt.grid(True)
    plt.colorbar(label='Küme No')
    plt.show()

    # Silhouette skorları için görselleştirme
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Silhouette Skoru')
    plt.title('Optimal Küme Sayısı Analizi')
    plt.grid(True)
    plt.show()

    # Sonuçları hazırlama
    results = {
        'clusters': [],
        'outliers': []
    }

    # Her küme için istatistikler
    for cluster in df['cluster'].unique():
        if cluster != -1:  # Aykırı değerleri hariç tut
            cluster_data = df[df['cluster'] == cluster]
            cluster_stats = {
                'cluster_number': int(cluster),
                'country_count': int(len(cluster_data)),
                'avg_total_orders': float(cluster_data['total_orders'].mean()),
                'avg_order_value': float(cluster_data['avg_order_value'].mean()),
                'avg_products_per_order': float(cluster_data['avg_products_per_order'].mean()),
                'countries': cluster_data[['country', 'total_orders', 'avg_order_value', 'avg_products_per_order']].to_dict('records')
            }
            results['clusters'].append(cluster_stats)

    # Aykırı değerleri (küme -1) analiz etme
    outliers = df[df['cluster'] == -1]
    results['outliers'] = outliers[['country', 'total_orders', 'avg_order_value', 'avg_products_per_order']].to_dict('records')

    return results

@app.get("/api/country-clustering", 
         response_model=CountryClusteringResponse,
         summary="Ülke satış deseni segmentasyonu sonuçlarını getir",
         description="Ülkeleri sipariş alışkanlıklarına göre analiz eder ve kümeleme sonuçlarını döndürür")
async def get_country_clustering_results():
    try:
        results = perform_country_clustering()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003) 