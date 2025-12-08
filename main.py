import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import xgboost as xgb
from datetime import datetime, time
import json
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="NSYSU violation risk predictor",
    layout="wide"
)

# ============================
# 1. 載入資料與模型
# ============================
@st.cache_data
def load_data():
    """載入區域資訊與座標資料"""
    try:
        # 載入區域資訊
        zone_info_path = 'data/zone_info.csv'
        if not os.path.exists(zone_info_path):
            zone_info_path = 'zone_info.csv'
        df_zones = pd.read_csv(zone_info_path, encoding='utf-8-sig')
        
        # 載入地點座標
        coords_path = 'data/unique_locations.csv'
        if not os.path.exists(coords_path):
            coords_path = 'unique_locations.csv'
        df_coords = pd.read_csv(coords_path, encoding='utf-8-sig')
        
        # 載入區域規則
        rules_path = 'data/location_rules.csv'
        if not os.path.exists(rules_path):
            rules_path = 'location_rules.csv'
        df_rules = pd.read_csv(rules_path, encoding='utf-8-sig')
        
        return df_zones, df_coords, df_rules
    except Exception as e:
        st.error(f"載入資料時發生錯誤: {e}")
        return None, None, None

@st.cache_resource
def load_model():
    """載入 XGBoost 模型"""
    try:
        model_path = 'parking_risk_model_v4_zone.json'
        if not os.path.exists(model_path):
            model_path = 'parking_risk_model_v4_zone.json'
        model = xgb.Booster()
        model.load_model(model_path)
        return model
    except Exception as e:
        st.warning(f"無法載入模型: {e}")
        return None

# ============================
# 2. 載入歷史資料用於特徵計算
# ============================
@st.cache_data
def load_historical_data():
    """載入歷史違規資料用於特徵計算"""
    try:
        violate_path = 'data/violate_with_type.csv'
        if not os.path.exists(violate_path):
            violate_path = 'violate_with_type.csv'
        df_raw = pd.read_csv(violate_path, encoding='utf-8-sig')
        df_raw['Datetime'] = pd.to_datetime(df_raw['舉發日期'], errors='coerce')
        df_raw = df_raw.dropna(subset=['Datetime'])
        return df_raw
    except Exception as e:
        st.warning(f"無法載入歷史資料: {e}")
        return None

@st.cache_data
def load_test_predictions():
    """載入測試集的預測結果（包含完整特徵）"""
    try:
        # 嘗試載入 notebook 輸出的測試集預測結果
        meta_test_path = 'meta_test.csv'
        if not os.path.exists(meta_test_path):
            return None
        
        df_test = pd.read_csv(meta_test_path, encoding='utf-8-sig')
        df_test['Slot_Start'] = pd.to_datetime(df_test['Slot_Start'])
        return df_test
    except Exception as e:
        st.warning(f"無法載入測試集預測結果: {e}")
        return None

@st.cache_data
def compute_zone_statistics(df_raw, df_rules):
    """計算區域的統計特徵"""
    if df_raw is None:
        return {}
    
    # 創建副本避免警告
    df_work = df_raw.copy()
    
    # 地點到區域映射
    loc_to_zone = df_rules.set_index('Original_Location')['Zone_ID'].to_dict()
    df_work['Zone_ID'] = df_work['違規地點'].map(loc_to_zone)
    df_work = df_work.dropna(subset=['Zone_ID'])
    df_work['Zone_ID'] = df_work['Zone_ID'].astype(int)
    
    # 計算區域基準風險
    zone_total = df_work.groupby('Zone_ID').size()
    zone_baseline_risk = (zone_total / zone_total.sum()).to_dict()
    
    # 計算上午/下午比例
    df_work['is_morning'] = ((df_work['Datetime'].dt.hour >= 9) & (df_work['Datetime'].dt.hour < 12)).astype(int)
    morning_counts = df_work[df_work['is_morning'] == 1].groupby('Zone_ID').size()
    afternoon_counts = df_work[df_work['is_morning'] == 0].groupby('Zone_ID').size()
    total_counts = df_work.groupby('Zone_ID').size()
    
    zone_morning_ratio = (morning_counts / total_counts).fillna(0.5).to_dict()
    zone_afternoon_ratio = (afternoon_counts / total_counts).fillna(0.5).to_dict()
    
    # 計算每個星期幾的風險
    df_work['weekday'] = df_work['Datetime'].dt.dayofweek
    zone_weekday_avg = df_work.groupby(['Zone_ID', 'weekday']).size().unstack(fill_value=0)
    zone_weekday_avg = zone_weekday_avg.div(zone_weekday_avg.sum(axis=1) + 1e-10, axis=0)
    
    # 計算星期幾+小時的風險
    df_work['hour'] = df_work['Datetime'].dt.hour
    zone_weekday_hour = df_work.groupby(['Zone_ID', 'weekday', 'hour']).size().reset_index(name='hist_count')
    zone_weekday_hour_total = df_work.groupby(['weekday', 'hour']).size().reset_index(name='total_count')
    zone_weekday_hour = zone_weekday_hour.merge(zone_weekday_hour_total, on=['weekday', 'hour'])
    zone_weekday_hour['zone_weekday_hour_rate'] = zone_weekday_hour['hist_count'] / (zone_weekday_hour['total_count'] + 1)
    
    return {
        'zone_baseline_risk': zone_baseline_risk,
        'zone_morning_ratio': zone_morning_ratio,
        'zone_afternoon_ratio': zone_afternoon_ratio,
        'zone_weekday_avg': zone_weekday_avg,
        'zone_weekday_hour': zone_weekday_hour
    }

# ============================
# 3. 特徵工程函數
# ============================
def create_features_for_prediction(zone_id, selected_datetime, zone_stats):
    """為單一區域和時間點創建完整特徵向量"""
    
    # 基礎時間特徵
    weekday = selected_datetime.weekday()
    hour = selected_datetime.hour
    minute = selected_datetime.minute
    
    features = {
        # 基礎時間特徵
        'weekday': weekday,
        'hour': hour,
        'minute': minute,
        'is_morning_session': 1 if hour < 12 else 0,
        'is_afternoon_session': 1 if hour >= 12 else 0,
        
        # 週期性編碼
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'weekday_sin': np.sin(2 * np.pi * weekday / 7),
        'weekday_cos': np.cos(2 * np.pi * weekday / 7),
        
        # Session progress
        'session_progress': 0.0,
        
        # 區域靜態特徵
        'zone_baseline_risk': 0.1,
        'zone_morning_ratio': 0.5,
        'zone_afternoon_ratio': 0.5,
        'zone_weekday_risk': 0.14,
        
        # 歷史特徵 (即時預測時設為預設值)
        'lag_1': 0.0,
        'lag_2': 0.0,
        'lag_3': 0.0,
        'lag_4': 0.0,
        'recent_1h_count': 0.0,
        'decay_recent': 0.0,
        'today_cumsum': 0.0,
        'today_global_cumsum': 0.0,
        'time_since_last_violation': 50.0,  # 預設中等值
        'zone_yesterday_count': 0.0,
        
        # 空間特徵 (即時預測時設為預設值)
        'neighbor_lag1_sum': 0.0,
        'neighbor_lag1_mean': 0.0,
        'neighbor_has_event': 0.0,
        'neighbor_event_count': 0.0,
        'neighbor_today_sum': 0.0,
        'neighbor_today_max': 0.0,
        
        # 進階特徵
        'zone_weekday_hour_rate': 0.0,
        'hist_slot_count': 0.0,
        
        # 交互特徵
        'risk_x_morning': 0.0,
        'risk_x_afternoon': 0.0,
        'risk_x_progress': 0.0,
        'weekday_hour_risk': 0.0,
        
        # 外部特徵
        'Precipitation': 0.0,
        'temperature': 25.0  # 預設溫度
    }
    
    # 計算 session_progress
    h, m = hour, minute
    if h < 12:
        start_min, end_min = 9 * 60, 11 * 60 + 30
    else:
        start_min, end_min = 14 * 60, 16 * 60 + 30
    current_min = h * 60 + m
    features['session_progress'] = np.clip((current_min - start_min) / (end_min - start_min), 0, 1)
    
    # 從統計資料中獲取區域特徵
    if zone_stats:
        features['zone_baseline_risk'] = zone_stats['zone_baseline_risk'].get(zone_id, 0.1)
        features['zone_morning_ratio'] = zone_stats['zone_morning_ratio'].get(zone_id, 0.5)
        features['zone_afternoon_ratio'] = zone_stats['zone_afternoon_ratio'].get(zone_id, 0.5)
        
        # 區域星期幾風險
        if zone_id in zone_stats['zone_weekday_avg'].index:
            features['zone_weekday_risk'] = zone_stats['zone_weekday_avg'].loc[zone_id, weekday]
        
        # 區域星期幾+小時風險率
        zone_wh = zone_stats['zone_weekday_hour']
        match = zone_wh[(zone_wh['Zone_ID'] == zone_id) & 
                        (zone_wh['weekday'] == weekday) & 
                        (zone_wh['hour'] == hour)]
        if not match.empty:
            features['zone_weekday_hour_rate'] = match.iloc[0]['zone_weekday_hour_rate']
    
    # 計算交互特徵
    features['risk_x_morning'] = features['zone_baseline_risk'] * features['is_morning_session']
    features['risk_x_afternoon'] = features['zone_baseline_risk'] * features['is_afternoon_session']
    features['risk_x_progress'] = features['zone_baseline_risk'] * features['session_progress']
    features['weekday_hour_risk'] = features['zone_weekday_risk'] * features['zone_weekday_hour_rate']
    
    return features

# ============================
# 4. 風險等級判定
# ============================
def get_risk_level(risk_score):
    """根據風險分數判定等級"""
    if risk_score >= 0.7:
        return "極高", "#8B0000"  # 深紅色
    elif risk_score >= 0.5:
        return "高", "#FF4444"    # 紅色
    elif risk_score >= 0.3:
        return "中", "#FFA500"    # 橙色
    elif risk_score >= 0.15:
        return "低", "#FFD700"    # 金色
    else:
        return "極低", "#90EE90"  # 淺綠色

# ============================
# 5. 地圖繪製
# ============================
def create_risk_map(df_zones, df_coords, df_rules, predictions, selected_datetime, map_mode='heatmap'):
    """創建風險地圖
    
    Args:
        map_mode: 'heatmap' 熱區圖模式 或 'marker' 標記模式
    """
    
    # 計算每個區域的中心座標
    coords_dict = df_coords.set_index('Location')[['Latitude', 'Longitude']].to_dict('index')
    zone_centers = {}
    
    for zone_id in df_zones['Zone_ID'].unique():
        zone_locs = df_rules[df_rules['Zone_ID'] == zone_id]['Original_Location'].tolist()
        lats = [coords_dict[loc]['Latitude'] for loc in zone_locs if loc in coords_dict]
        lons = [coords_dict[loc]['Longitude'] for loc in zone_locs if loc in coords_dict]
        if lats and lons:
            zone_centers[zone_id] = {
                'lat': np.mean(lats),
                'lon': np.mean(lons),
                'name': df_zones[df_zones['Zone_ID'] == zone_id]['Zone_Name'].iloc[0]
            }
    
    # 創建地圖 (中山大學中心)
    m = folium.Map(
        location=[22.6273, 120.2659],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    if map_mode == 'heatmap':
        # ========== 熱區圖模式 ==========
        # 準備熱力圖資料: [lat, lon, weight]
        heat_data = []

        for zone_id, pred_data in predictions.items():
            if zone_id not in zone_centers:
                continue

            center = zone_centers[zone_id]
            raw_score = float(pred_data['risk_score'])

            # 將風險分數限制在 (0,1) 範圍內，並做一次非線性壓縮
            risk_score = float(np.clip(raw_score, 0.01, 0.99))
            risk_score = risk_score ** 0.7

            # 每個區只加一個點，由 HeatMap 的 radius/blur 決定範圍
            heat_data.append([
                center['lat'],
                center['lon'],
                risk_score
            ])
        
        # 添加熱力圖層：較大的 radius / blur 讓每個區呈現一整塊範圍
        HeatMap(
            heat_data,
            min_opacity=0.2,
            max_opacity=0.9,
            radius=35,  # 區域範圍大小
            blur=30,    # 範圍邊界柔和程度
            max_val=1.0,
            gradient={
                0.0: '#FFFFFF',
                0.2: '#90EE90',
                0.4: '#FFFF99',
                0.6: '#FFA500',
                0.8: '#FF4444',
                1.0: '#8B0000'
            }
        ).add_to(m)
        
        # 添加區域標籤和資訊標記
        for zone_id, pred_data in predictions.items():
            if zone_id not in zone_centers:
                continue
                
            center = zone_centers[zone_id]
            risk_score = pred_data['risk_score']
            risk_level, color = get_risk_level(risk_score)
            
            # 創建彈出視窗
            popup_html = f"""
            <div style="font-family: Arial; min-width: 200px;">
                <h4 style="margin-bottom: 10px;">{center['name']}</h4>
                <p><b>風險等級:</b> <span style="color: {color}; font-weight: bold;">{risk_level}</span></p>
                <p><b>風險分數:</b> {risk_score:.2%}</p>
                <p><b>時間:</b> {selected_datetime.strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            """
            
            # 小圓點標記位置
            folium.CircleMarker(
                location=[center['lat'], center['lon']],
                radius=5,
                popup=folium.Popup(popup_html, max_width=300),
                color='white',
                fill=True,
                fillColor='black',
                fillOpacity=0.6,
                weight=1
            ).add_to(m)
            
            # 添加區域名稱標籤
            folium.Marker(
                location=[center['lat'], center['lon']],
                icon=folium.DivIcon(html=f"""
                    <div style="font-size: 9pt; color: black; font-weight: bold; 
                                text-shadow: 1px 1px 3px white, -1px -1px 3px white,
                                            1px -1px 3px white, -1px 1px 3px white;
                                white-space: nowrap;">
                        {center['name']}
                    </div>
                """)
            ).add_to(m)
        
    else:
        # ========== 標記模式 (原始) ==========
        for zone_id, pred_data in predictions.items():
            if zone_id not in zone_centers:
                continue
                
            center = zone_centers[zone_id]
            risk_score = pred_data['risk_score']
            risk_level, color = get_risk_level(risk_score)
            
            popup_html = f"""
            <div style="font-family: Arial; min-width: 200px;">
                <h4 style="margin-bottom: 10px;">{center['name']}</h4>
                <p><b>風險等級:</b> <span style="color: {color}; font-weight: bold;">{risk_level}</span></p>
                <p><b>風險分數:</b> {risk_score:.2%}</p>
                <p><b>時間:</b> {selected_datetime.strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            """
            
            radius = 10 + risk_score * 30
            
            folium.CircleMarker(
                location=[center['lat'], center['lon']],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
            
            folium.Marker(
                location=[center['lat'], center['lon']],
                icon=folium.DivIcon(html=f"""
                    <div style="font-size: 10pt; color: black; font-weight: bold; 
                                text-shadow: 1px 1px 2px white, -1px -1px 2px white;">
                        {center['name']}
                    </div>
                """)
            ).add_to(m)
    
    # 添加圖例
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: 220px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:13px; padding: 10px; border-radius: 5px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
    <p style="margin-bottom: 10px; font-weight: bold; font-size: 15px;">🔥 風險熱區圖例</p>
    <div style="background: linear-gradient(to top, #FFFFFF, #90EE90, #FFFF99, #FFD700, #FFA500, #FF6347, #FF4444, #8B0000); 
                height: 120px; width: 35px; float: left; margin-right: 10px; border-radius: 3px; border: 1px solid #ccc;"></div>
    <div style="float: left; line-height: 16px; font-size: 12px;">
        <p style="margin: 0; color: #8B0000;">■ 極高 (≥70%)</p>
        <p style="margin: 12px 0 0 0; color: #FF4444;">■ 高 (50-70%)</p>
        <p style="margin: 12px 0 0 0; color: #FFA500;">■ 中 (30-50%)</p>
        <p style="margin: 12px 0 0 0; color: #FFD700;">■ 低-中 (20-30%)</p>
        <p style="margin: 12px 0 0 0; color: #FFFF99;">■ 低 (10-20%)</p>
        <p style="margin: 12px 0 0 0; color: #90EE90;">■ 極低 (<10%)</p>
    </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# ============================
# 6. 主程式
# ============================
def main():
    st.title("NSYSU violation risk predictor")
    st.markdown("---")
    
    # 載入資料
    df_zones, df_coords, df_rules = load_data()
    model = load_model()
    df_raw = load_historical_data()
    
    # 計算區域統計特徵
    zone_stats = compute_zone_statistics(df_raw, df_rules) if df_raw is not None else {}
    
    if df_zones is None or df_coords is None:
        st.error("無法載入必要資料，請確認資料檔案存在。")
        return
    
    # 側邊欄控制
    st.sidebar.header("Prediction Settings")
    
    # 地圖模式選擇
    map_mode = st.sidebar.radio(
        "Map Mode",
        ["Heatmap", "Marker Mode"],
        index=0
    )
    map_mode_value = 'heatmap' if 'Heatmap' in map_mode else 'marker'
    
    st.sidebar.markdown("---")
    
    # 日期選擇 (限制在測試集範圍)
    selected_date = st.sidebar.date_input(
        "Select Date",
        datetime(2025, 10, 15),  # 預設日期在測試集內
        min_value=datetime(2025, 5, 7),   # 測試集開始日期
        max_value=datetime(2025, 11, 21), # 測試集結束日期
        help="Test set date range (2025-05-07 ~ 2025-11-21)"
    )
    
    # 時間選擇
    time_slot = st.sidebar.radio(
        "Select Time Slot",
        ["Morning (09:00-11:30)", "Afternoon (14:00-16:30)"]
    )
    
    if "上午" in time_slot:
        default_time = time(10, 0)
        time_options = [(9 + h, m) for h in range(3) for m in [0, 15, 30, 45] 
                       if not (h == 2 and m > 30)]
    else:
        default_time = time(15, 0)
        time_options = [(14 + h, m) for h in range(3) for m in [0, 15, 30, 45] 
                       if not (h == 2 and m > 30)]
    
    selected_time = st.sidebar.time_input(
        "Select Time",
        default_time
    )
    
    # 合併日期時間
    selected_datetime = datetime.combine(selected_date, selected_time)
    
    # 顯示選擇的時間
    st.sidebar.markdown("---")
    st.sidebar.info(f"Prediction Time:\n\n**{selected_datetime.strftime('%Y-%m-%d %H:%M')}**")
    
    # 載入測試集預測結果
    df_test_predictions = load_test_predictions()
    
    # 預測按鈕
    if st.sidebar.button("Start Prediction", type="primary"):
        
        with st.spinner("Analyzing zone risks..."):
            predictions = {}
            use_test_data = False
            
            # 將選擇的時間對齊到 15 分鐘
            selected_datetime_aligned = selected_datetime.replace(second=0, microsecond=0)
            minute = selected_datetime_aligned.minute
            selected_datetime_aligned = selected_datetime_aligned.replace(
                minute=(minute // 15) * 15
            )
            
            # 檢查是否可以從測試集獲取數據
            if df_test_predictions is not None:
                test_data_this_slot = df_test_predictions[
                    df_test_predictions['Slot_Start'] == selected_datetime_aligned
                ]
                
                if not test_data_this_slot.empty:
                    use_test_data = True
                    st.sidebar.success("Using test set predictions")
                    
                    # 從測試集直接獲取預測結果
                    for _, row in test_data_this_slot.iterrows():
                        zone_id = int(row['Zone_ID'])
                        zone_name = df_zones[df_zones['Zone_ID'] == zone_id]['Zone_Name'].iloc[0] if zone_id in df_zones['Zone_ID'].values else f'Zone_{zone_id}'
                        
                        predictions[zone_id] = {
                            'zone_name': zone_name,
                            'risk_score': float(row['risk_score_xgb']),
                            'risk_level': get_risk_level(float(row['risk_score_xgb']))[0],
                            'actual_label': int(row['label']) if 'label' in row else None
                        }
                else:
                    st.sidebar.info("Time not in test set, using real-time prediction")
            
            # 如果無法使用測試集數據，則使用實時預測
            if not use_test_data:
                # 特徵順序 (必須與訓練時一致)
                feature_names = [
                    'weekday', 'hour', 'minute',
                    'is_morning_session', 'is_afternoon_session',
                    'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
                    'session_progress',
                    'zone_baseline_risk', 'zone_morning_ratio', 'zone_afternoon_ratio', 'zone_weekday_risk',
                    'lag_1', 'lag_2', 'lag_3', 'lag_4',
                    'recent_1h_count', 'decay_recent',
                    'today_cumsum', 'today_global_cumsum', 'time_since_last_violation', 'zone_yesterday_count',
                    'neighbor_lag1_sum', 'neighbor_lag1_mean', 'neighbor_has_event', 'neighbor_event_count',
                    'neighbor_today_sum', 'neighbor_today_max',
                    'zone_weekday_hour_rate', 'hist_slot_count',
                    'risk_x_morning', 'risk_x_afternoon', 'risk_x_progress', 'weekday_hour_risk',
                    'Precipitation', 'temperature'
                ]
                
                # 為每個區域預測
                for _, row in df_zones.iterrows():
                    zone_id = row['Zone_ID']
                    
                    # 創建特徵
                    features_dict = create_features_for_prediction(zone_id, selected_datetime_aligned, zone_stats)
                    
                    # 使用真實模型預測
                    if model is not None:
                        try:
                            # 按照正確順序排列特徵 (與 notebook 完全一致)
                            feature_values = [features_dict[feat] for feat in feature_names]
                            X = pd.DataFrame([feature_values], columns=feature_names)
                            dmatrix = xgb.DMatrix(X)
                            risk_score = float(model.predict(dmatrix)[0])
                            
                            # Debug: 顯示實際使用模型
                            if zone_id == 1:  # 只為第一個區域顯示一次
                                st.sidebar.success("Using trained XGBoost model")
                                
                        except Exception as e:
                            st.error(f"Zone {zone_id} 預測失敗: {e}")
                            st.error(f"特徵數量: {len(feature_values)}, 預期: {len(feature_names)}")
                            # 使用基於統計的備用預測
                            base_risk = features_dict['zone_baseline_risk']
                            time_factor = features_dict['session_progress'] * 0.2
                            risk_score = np.clip(base_risk + time_factor, 0, 1)
                    else:
                        st.sidebar.warning("Model not loaded, using statistical estimation")
                        # 無模型時使用統計估算
                        base_risk = features_dict['zone_baseline_risk']
                        session_factor = features_dict['session_progress'] * 0.3
                        weekday_factor = features_dict['zone_weekday_risk'] * 0.5
                        risk_score = np.clip(base_risk + session_factor + weekday_factor, 0, 1)
                    
                    predictions[zone_id] = {
                        'zone_name': row['Zone_Name'],
                        'risk_score': risk_score,
                        'risk_level': get_risk_level(risk_score)[0],
                        'actual_label': None
                    }
            
            # 顯示統計資訊
            st.subheader(f"Prediction Time: {selected_datetime_aligned.strftime('%Y-%m-%d %H:%M')}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            risk_scores = [p['risk_score'] for p in predictions.values()]
            
            with col1:
                st.metric("Average Risk", f"{np.mean(risk_scores):.1%}")
            with col2:
                high_risk = sum(1 for s in risk_scores if s >= 0.5)
                st.metric("High Risk Zones", f"{high_risk}")
            with col3:
                safe_zones = sum(1 for s in risk_scores if s < 0.3)
                st.metric("Safe Zones", f"{safe_zones}")
            with col4:
                st.metric("Prediction Mode", "Test Set" if use_test_data else "Real-time")
            
            st.markdown("---")
            
            # 顯示地圖
            if map_mode_value == 'heatmap':
                st.subheader("Risk Heatmap")
                st.caption("Redder colors indicate higher violation risk. Click black dots for zone details.")
            else:
                st.subheader("Risk Distribution Map")
                st.caption("Circle size and color indicate risk level. Click for details.")
            
            risk_map = create_risk_map(df_zones, df_coords, df_rules, predictions, selected_datetime, map_mode_value)
            folium_static(risk_map, width=1200, height=600)
            
            # 顯示排名和表格
            st.markdown("---")
            
            # 創建完整結果 DataFrame
            results_df = pd.DataFrame([
                {
                    'Rank': idx,
                    'Zone': data['zone_name'],
                    'Risk Score': data['risk_score'],
                    'Risk Level': data['risk_level']
                }
                for idx, (zone_id, data) in enumerate(
                    sorted(predictions.items(), key=lambda x: x[1]['risk_score'], reverse=True), 1
                )
            ])
            
            # 分兩欄顯示
            col_left, col_right = st.columns(2)
            
            # Top 10 危險區域
            with col_left:
                st.subheader("Top 10 High Risk Zones")
                top_10 = results_df.head(10).copy()
                
                # 為風險分數添加顏色
                def highlight_risk(row):
                    risk_score = row['Risk Score']
                    _, color = get_risk_level(risk_score)
                    return [f'background-color: {color}30' if col == 'Risk Score' else '' for col in row.index]
                
                styled_top = top_10.style.format({'Risk Score': '{:.2%}'}).apply(highlight_risk, axis=1)
                st.dataframe(styled_top, use_container_width=True, hide_index=True)
            
            # Bottom 10 安全區域
            with col_right:
                st.subheader("Bottom 10 Safe Zones")
                bottom_10 = results_df.tail(10).iloc[::-1].copy()
                bottom_10['Rank'] = range(len(results_df), len(results_df) - 10, -1)
                
                styled_bottom = bottom_10.style.format({'Risk Score': '{:.2%}'}).apply(highlight_risk, axis=1)
                st.dataframe(styled_bottom, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # 添加風險分布圖表
            st.subheader("Risk Score Distribution")
            
            import plotly.graph_objects as go
            import plotly.express as px
            
            fig = go.Figure()
            
            # 添加柱狀圖
            colors = [get_risk_level(row['Risk Score'])[1] for _, row in results_df.iterrows()]
            
            fig.add_trace(go.Bar(
                x=results_df['Zone'],
                y=results_df['Risk Score'],
                marker_color=colors,
                text=results_df['Risk Score'].apply(lambda x: f'{x:.1%}'),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Risk Score: %{y:.2%}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Risk Score Ranking by Zone',
                xaxis_title='Zone',
                yaxis_title='Risk Score',
                yaxis=dict(tickformat='.0%'),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
    
    else:
        # 顯示說明
        st.info("Please configure prediction parameters on the left sidebar and click 'Start Prediction' button")
        
        st.markdown("""
        ## User Guide
        
        This system uses machine learning models to predict parking violation enforcement risk across NSYSU campus zones.
        
        ### Features
        - **Visual Map**: Intuitive risk distribution visualization
        - **Ranking Recommendations**: Identifies most dangerous and safest zones
        
        ### Model Performance
        - **AUC-ROC**: 0.82 (Excellent)
        - **Hit Rate@5**: 80.55% (5-zone recommendation accuracy)
        - **Lift Score**: 3.71x (Risk identification improvement)
        
        ### Risk Level Definitions
        - **Extreme/High**: Strongly recommend avoiding
        - **Medium**: Park with caution
        - **Low**: Relatively safe
        - **Very Low**: Recommended parking
        
        """)
        
        # 顯示示範圖片或統計
        st.markdown("---")
        st.subheader("System Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Coverage", "22 Zones")
        with col2:
            st.metric("Data Period", "2023-2025")
        with col3:
            st.metric("Accuracy", "82%")

if __name__ == "__main__":
    main()