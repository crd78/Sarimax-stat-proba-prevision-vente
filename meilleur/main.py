import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.2)

def load_and_preprocess_data(file_path):
    print("Loading and preprocessing data...")

    # Lire le CSV StatCan avec le bon séparateur
    df = pd.read_csv(file_path, sep=';', dtype=str)

    # Nettoyer les noms de colonnes
    df.columns = [col.replace('"', '').replace("'", '').strip() for col in df.columns]
    print("Colonnes après nettoyage :", list(df.columns))

    # Trouver la colonne de date (souvent la première)
    date_col = [col for col in df.columns if 'PÉRIODE' in col.upper() or 'REFERENCE' in col.upper()][0]
    df['Date'] = pd.to_datetime(df[date_col], format='%Y-%m', errors='coerce')

    # Nettoyer la colonne valeur
    df['Montant'] = (
        df['VALEUR']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .str.replace(' ', '', regex=False)
        .astype(float)
    )

    # Produit = colonne industrie
    produit_col = [col for col in df.columns if 'classification' in col.lower() or 'industrie' in col.lower()][0]
    df['Produit'] = df[produit_col]

    # Trier par date
    df = df.sort_values('Date')

    # Ventes mensuelles (tous produits)
    monthly_sales = df.groupby('Date')['Montant'].sum().reset_index()
    monthly_sales = monthly_sales.set_index('Date').resample('M').sum()

    # Ventes mensuelles par produit
    product_monthly = (
        df.groupby(['Date', 'Produit'])['Montant'].sum().reset_index()
        .set_index(['Date', 'Produit'])
        .groupby(level='Produit')
        .resample('M', level='Date')
        .sum()
        .reset_index()
    )

    return df, monthly_sales, product_monthly

def visualize_data(df, monthly_sales, product_monthly):
    print("Creating visualizations...")

    fig = plt.figure(figsize=(20, 15))

    # Plot 1: Monthly sales trend
    ax1 = plt.subplot(2, 1, 1)
    monthly_sales.plot(ax=ax1, title='Monthly Sales Trend', legend=False, color='blue')
    ax1.set_ylabel('Sales Amount')
    ax1.grid(True)

    # Plot 2: Product comparison (top 5 products)
    ax2 = plt.subplot(2, 1, 2)
    top_products = df.groupby('Produit')['Montant'].sum().nlargest(5).index
    for product in top_products:
        product_data = product_monthly[product_monthly['Produit'] == product]
        ax2.plot(product_data['Date'], product_data['Montant'], label=str(product))
    ax2.set_title('Monthly Sales for Top 5 Products')
    ax2.set_ylabel('Sales Amount')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('sales_analysis.png')
    plt.close()

    print("Visualizations saved to sales_analysis.png")

def build_and_evaluate_model(monthly_sales):
    """Build and evaluate a SARIMA model for forecasting with optimized parameters"""
    print("Building optimized forecasting model...")
    
    # Préparation des données d'entraînement
    train_data = monthly_sales
    
  
    best_aic = float("inf")
    best_params = None
    best_results = None
    
   
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    P_values = [0, 1]
    D_values = [0, 1]
    Q_values = [0, 1]
    s_values = [12] 
    
    print("Recherche des meilleurs paramètres pour le modèle...")
    
    # Utiliser un sous-ensemble des combinaisons pour accélérer le processus
    for p in p_values[:2]:  # Réduire la portée de la recherche
        for d in d_values:
            for q in q_values[:2]:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            for s in s_values:
                                try:
                                    model = SARIMAX(
                                        train_data,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, s),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )
                                    results = model.fit(disp=False)
                                    aic = results.aic
                                    
                              
                                    if aic < best_aic:
                                        best_aic = aic
                                        best_params = (p, d, q, P, D, Q, s)
                                        best_results = results
                                    
                                except:
                                    continue
    
    if best_results is None:
        print("Échec de l'optimisation, utilisation de paramètres par défaut")
        model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                       enforce_stationarity=False, enforce_invertibility=False)
        best_results = model.fit(disp=False)
    else:
        print(f"Meilleurs paramètres SARIMA: ordre={best_params[:3]}, ordre saisonnier={best_params[3:]}")
    
    # Évaluation du modèle sur des données historiques
    if len(monthly_sales) > 3:
        start_loc = len(monthly_sales) // 2  # Utiliser le milieu de nos données
        start_date = monthly_sales.index[start_loc]
        
        pred = best_results.get_prediction(start=start_date, dynamic=False)
        pred_ci = pred.conf_int()
        
        # Extraire les prédictions moyennes
        y_pred = pred.predicted_mean
        
        # Calculer les métriques d'erreur
        overlap_data = monthly_sales.loc[y_pred.index]
        if len(overlap_data) > 0:
            mae = mean_absolute_error(overlap_data, y_pred)
            rmse = np.sqrt(mean_squared_error(overlap_data, y_pred))
            print(f"Métriques d'évaluation du modèle:")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
    
    return best_results

def forecast_future_sales(model_results, months_to_forecast=12, conf_int=0.80):
    """Generate sales forecasts for future months"""
    print(f"Forecasting sales for next {months_to_forecast} months...")
    
    
    forecast = model_results.get_forecast(steps=months_to_forecast)
    forecast_ci = forecast.conf_int(alpha=1-conf_int)  # alpha=0.2 donne un intervalle de confiance de 80%
    

    last_date = model_results.model.data.dates[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months_to_forecast, freq='M')
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast.predicted_mean.values,
        'Lower CI': forecast_ci.iloc[:, 0].values,
        'Upper CI': forecast_ci.iloc[:, 1].values
    })
    

    plt.figure(figsize=(12, 6))
    

    historical_dates = model_results.model.data.dates
    historical_values = model_results.model.data.endog
    plt.plot(historical_dates, historical_values, label='Ventes Historiques')

    plt.plot(forecast_df['Date'], forecast_df['Forecast'], color='red', label='Prévision')
    plt.fill_between(forecast_df['Date'], forecast_df['Lower CI'], forecast_df['Upper CI'], 
                    color='pink', alpha=0.3, label=f'Intervalle de confiance {int(conf_int*100)}%')
    
    plt.title('Prévision des Ventes')
    plt.xlabel('Date')
    plt.ylabel('Montant des Ventes')
    plt.legend()
    plt.grid(True)
    plt.savefig('sales_forecast.png')
    plt.close()
    
 
    forecast_df.to_csv('sales_forecast.csv', index=False)
    
    print("Forecast saved to sales_forecast.csv and sales_forecast.png")
    return forecast_df

def product_specific_forecasts(product_monthly, top_n=5, conf_int=0.90):
    print(f"Generating optimized product-specific forecasts for top {top_n} products...")

    top_products = product_monthly.groupby('Produit')['Montant'].sum().nlargest(top_n).index

    plt.figure(figsize=(15, 10))
    all_product_forecasts = {}

    for i, product in enumerate(top_products, 1):
        print(f"Forecasting for {product}...")
        product_data = product_monthly[product_monthly['Produit'] == product].copy()
        product_data = product_data.set_index('Date')['Montant']
        idx = pd.date_range(product_data.index.min(), product_data.index.max(), freq='M')
        product_ts = product_data.reindex(idx).fillna(product_data.mean())

        try:
            best_aic = float("inf")
            best_params = None
            best_model = None
            p_params = [0, 1]
            d_params = [1]
            q_params = [0, 1]
            for p in p_params:
                for d in d_params:
                    for q in q_params:
                        try:
                            model = SARIMAX(
                                product_ts,
                                order=(p, d, q),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_invertibility=False
                            )
                            results = model.fit(disp=False)
                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_params = (p, d, q)
                                best_model = results
                        except:
                            continue
            if best_model is None:
                model = SARIMAX(product_ts, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False, enforce_invertibility=False)
                results = model.fit(disp=False)
            else:
                print(f"  Meilleurs paramètres pour {product}: {best_params}")
                results = best_model

            forecast = results.get_forecast(steps=6)
            forecast_ci = forecast.conf_int(alpha=1-conf_int)
            last_date = results.model.data.dates[-1]
            forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='M')
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecast.predicted_mean.values,
                'Lower CI': forecast_ci.iloc[:, 0].values,
                'Upper CI': forecast_ci.iloc[:, 1].values
            })
            all_product_forecasts[product] = forecast_df
            plt.subplot(len(top_products), 1, i)
            plt.plot(product_ts.index, product_ts.values, label='Historique')
            plt.plot(forecast_df['Date'], forecast_df['Forecast'], color='red', label='Prévision')
            plt.fill_between(forecast_df['Date'], forecast_df['Lower CI'], forecast_df['Upper CI'],
                           color='pink', alpha=0.3, label=f'IC {int(conf_int*100)}%')
            plt.title(f'Prévision optimisée pour {product}')
            plt.legend()
            plt.grid(True)
        except Exception as e:
            print(f"Erreur lors de la prévision pour {product}: {e}")

    plt.tight_layout()
    plt.savefig('product_forecasts_optimized.png')
    plt.close()

    for product, forecast in all_product_forecasts.items():
        forecast.to_csv(f'forecast_{str(product).replace(" ", "_")}_optimized.csv', index=False)

    print("Prévisions par produit optimisées enregistrées")
    return all_product_forecasts

def main():
    """Main function to orchestrate the optimized sales forecasting process"""
    print("Starting optimized sales forecasting process...")
    
 
    file_path = 'meilleur/20100082.csv'
    
 
    df, monthly_sales, product_monthly = load_and_preprocess_data(file_path)
    
   
    visualize_data(df, monthly_sales, product_monthly)
    
  
    model_results = build_and_evaluate_model(monthly_sales)
    

    forecast_df = forecast_future_sales(model_results, months_to_forecast=12, conf_int=0.90)
    
 
    product_forecasts = product_specific_forecasts(product_monthly, conf_int=0.90)
    
    print("Optimized sales forecasting process completed successfully!")
    return forecast_df, product_forecasts

if __name__ == "__main__":
    main()