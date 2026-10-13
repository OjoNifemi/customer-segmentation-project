import pandas as pd
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import KMeans  
import joblib  
from config import FEATURE_NAMES, SCALER_PATH, KMEANS_PATH,BASE_DIR


df = pd.read_csv(BASE_DIR / "data" / "Mall_Customers.csv")
X = df[FEATURE_NAMES]

scaler = StandardScaler()

df_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=6, n_init=10, random_state=42)
kmeans.fit(df_scaled)

unscaled_centers = scaler.inverse_transform(kmeans.cluster_centers_)

unscaled_centers_df = pd.DataFrame(unscaled_centers, columns=X.columns)

print("Cluster Centers (Unscaled):")
print(unscaled_centers_df)


KMEANS_PATH.parent.mkdir(exist_ok=True)
joblib.dump(kmeans, KMEANS_PATH)
joblib.dump(scaler, SCALER_PATH)
