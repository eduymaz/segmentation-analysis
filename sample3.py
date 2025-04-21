# Tedarikçi Segmentasyonu
# DBSCAN kullanarak tedarikçileri satış performanslarına göre gruplandırma
# Az katkı sağlayan veya sıra dışı tedarikçileri belirleme

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

app = FastAPI(title="Tedarikçi Segmentasyonu API",
             description="Tedarikçileri satış performanslarına göre kümeleme ve analiz etme API'si",
             version="1.0.0")

class SupplierStats(BaseModel):
    supplier_id: int
    supplier_name: str
    product_count: int
    total_sales: float
    avg_price: float
    unique_customers: int
    cluster: int

class SupplierClusteringResponse(BaseModel):
    clusters: List[Dict[str, Any]]
    outliers: List[Dict[str, Any]]

def perform_supplier_clustering():
    # Veritabanı bağlantısı
    user = 'postgres'
    password = "1352"
    host = 'localhost'
    port = '5432'
    database = 'nortwind'

    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}") 

    # Tedarikçi verilerini çekme
    query = """
    SELECT 
        s.supplier_id,
        s.company_name as supplier_name,
        COUNT(DISTINCT p.product_id) as product_count,
        SUM(od.quantity) as total_sales,
        AVG(od.unit_price) as avg_price,
        COUNT(DISTINCT o.customer_id) as unique_customers
    FROM 
        suppliers s
    INNER JOIN 
        products p ON s.supplier_id = p.supplier_id
    INNER JOIN 
        order_details od ON p.product_id = od.product_id
    INNER JOIN 
        orders o ON od.order_id = o.order_id
    GROUP BY 
        s.supplier_id, s.company_name
    HAVING 
        COUNT(DISTINCT p.product_id) > 0
    """

    df = pd.read_sql_query(query, engine)

    # Özellik seçimi ve ölçeklendirme
    features = ['product_count', 'total_sales', 'avg_price', 'unique_customers']
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
    plt.scatter(df['total_sales'], df['product_count'], c=df['cluster'], cmap='viridis', s=60)
    plt.xlabel("Toplam Satış Miktarı")
    plt.ylabel("Tedarik Edilen Ürün Sayısı")
    plt.title("Tedarikçi Segmentasyonu (DBSCAN)")
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
                'supplier_count': int(len(cluster_data)),
                'avg_product_count': float(cluster_data['product_count'].mean()),
                'avg_total_sales': float(cluster_data['total_sales'].mean()),
                'avg_price': float(cluster_data['avg_price'].mean()),
                'suppliers': cluster_data[['supplier_id', 'supplier_name', 'product_count', 'total_sales', 'avg_price', 'unique_customers']].to_dict('records')
            }
            results['clusters'].append(cluster_stats)

    # Aykırı değerleri (küme -1) analiz etme
    outliers = df[df['cluster'] == -1]
    results['outliers'] = outliers[['supplier_id', 'supplier_name', 'product_count', 'total_sales', 'avg_price', 'unique_customers']].to_dict('records')

    return results

@app.get("/api/supplier-clustering", 
         response_model=SupplierClusteringResponse,
         summary="Tedarikçi segmentasyonu sonuçlarını getir",
         description="Tedarikçileri satış performanslarına göre analiz eder ve kümeleme sonuçlarını döndürür")
async def get_supplier_clustering_results():
    try:
        results = perform_supplier_clustering()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 