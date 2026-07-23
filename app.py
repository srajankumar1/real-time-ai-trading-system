import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.data_pipeline import download_historical_data
from src.feature_engineering import generate_technical_features, generate_target_labels
from src.model_engine import TimeSeriesEnsembleEngine
from src.backtester import run_vectorized_backtest
import asyncio
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="High-Efficiency AI Trading Core Engine")
MODEL_REGISTRY = {}

def train_single_asset(ticker, data):
    try:
        feat_df = generate_technical_features(data)
        labeled_df = generate_target_labels(feat_df)
        
        engine = TimeSeriesEnsembleEngine()
        X, y = engine.prepare_matrices(labeled_df)
        
        mean_cv = engine.train_walk_forward_validation(X, y)
        metrics, preds = engine.evaluate_model(X, y)
        fin_metrics, _ = run_vectorized_backtest(labeled_df['Close'], preds)
        
        return ticker, {
            "engine": engine,
            "last_row": labeled_df.get(engine.feature_cols).iloc[-1:],
            "raw_df": labeled_df.iloc[-1],
            "fin_metrics": fin_metrics,
            "cls_metrics": metrics,
            "cv_score": mean_cv
        }
    except Exception as e:
        print(f"Failed loading modeling matrix parameters for ticker {ticker}: {str(e)}")
        return ticker, None

@app.on_event("startup")
async def execute_system_initialization_sequence():
    global MODEL_REGISTRY
    raw_assets = download_historical_data()
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(executor, train_single_asset, ticker, data)
            for ticker, data in raw_assets.items()
        ]
        results = await asyncio.gather(*tasks)
        
    for ticker, result in results:
        if result is not None:
            MODEL_REGISTRY[ticker] = result
            print(f"System Optimized [Parallel]: {ticker} | Validation Score: {result['cv_score']:.2%}")

@app.websocket("/ws/signals")
async def telemetry_signals_broadcast_stream(websocket: WebSocket):
    await websocket.accept()
    print("UI connection pipeline verified.")
    try:
        while True:
            for ticker, components in MODEL_REGISTRY.items():
                engine = components["engine"]
                X_live = components["last_row"].values
                raw_row = components["raw_df"]
                fin_metrics = components["fin_metrics"]
                
                X_scaled = engine.scaler.transform(X_live)
                pred = int(engine.best_ensemble.predict(X_scaled)[0])
                prob = engine.best_ensemble.predict_proba(X_scaled)[0]
                confidence = float(np.max(prob))
                
                label_map = {0: "HOLD/WAIT", 1: "BUY", 2: "SELL"}
                
                payload = {
                    "ticker": ticker,
                    "signal": label_map[pred],
                    "confidence": f"{confidence:.2%}",
                    "rsi": float(raw_row['RSI']),
                    "close": float(raw_row['Close']),
                    "sharpe": f"{fin_metrics['sharpe_ratio']:.2f}",
                    "cagr": f"{fin_metrics['cagr']:.2f}%",
                    "max_dd": f"{fin_metrics['max_drawdown']:.2f}%"
                }
                await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        print("Frontend UI socket pipeline disconnected safely.")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)