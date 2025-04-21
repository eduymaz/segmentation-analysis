# Tüm Segmentasyon API'lerini Birleştirme
# Müşteri, Ürün, Tedarikçi ve Ülke Segmentasyonu API'leri

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Diğer dosyalardan fonksiyonları import et
from sample2 import perform_product_clustering
from sample3 import perform_supplier_clustering
from sample4 import perform_country_clustering

app = FastAPI(title="Segmentasyon API'leri",
             description="Müşteri, Ürün, Tedarikçi ve Ülke Segmentasyonu API'leri",
             version="1.0.0")

# Response modelleri
class ProductClusteringResponse(BaseModel):
    clusters: List[Dict[str, Any]]
    outliers: List[Dict[str, Any]]

class SupplierClusteringResponse(BaseModel):
    clusters: List[Dict[str, Any]]
    outliers: List[Dict[str, Any]]

class CountryClusteringResponse(BaseModel):
    clusters: List[Dict[str, Any]]
    outliers: List[Dict[str, Any]]

# API Endpoint'leri
@app.get("/api/product-clustering", 
         response_model=ProductClusteringResponse,
         summary="Ürün segmentasyonu sonuçlarını getir",
         description="Ürünleri satış performanslarına göre analiz eder ve kümeleme sonuçlarını döndürür")
async def get_product_clustering_results():
    try:
        results = perform_product_clustering()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    uvicorn.run(app, host="0.0.0.0", port=8000) 