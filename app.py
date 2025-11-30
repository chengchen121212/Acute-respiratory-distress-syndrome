"""
LightGBM风险预测模型Web应用
基于公共数据库训练的模型，用于医院数据的风险预测
"""
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# 加载模型
MODEL_PATH = 'LGBMClassifier8221655.pkl'
model = None

def load_model():
    """加载预训练的LightGBM模型"""
    global model
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"模型加载成功！特征数量: {model.n_features_}")
        print(f"特征名称: {model.feature_name_}")
        return True
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return False

# 特征定义和说明
FEATURE_INFO = {
    'admission_age': {
        'name': '入院年龄',
        'unit': '岁',
        'range': [0, 120],
        'description': '患者入院时的年龄'
    },
    'spo2': {
        'name': '血氧饱和度',
        'unit': '%',
        'range': [0, 100],
        'description': '动脉血氧饱和度'
    },
    'sofa': {
        'name': 'SOFA评分',
        'unit': '分',
        'range': [0, 24],
        'description': '序贯器官衰竭评分'
    },
    'admission_vent_mode': {
        'name': '入院通气模式',
        'unit': '',
        'range': [0, 10],
        'description': '入院时的机械通气模式编码'
    },
    'calcium': {
        'name': '血钙',
        'unit': 'mmol/L',
        'range': [0, 5],
        'description': '血清钙浓度'
    },
    'resp_rate': {
        'name': '呼吸频率',
        'unit': '次/分',
        'range': [0, 60],
        'description': '每分钟呼吸次数'
    },
    'gcs': {
        'name': 'GCS评分',
        'unit': '分',
        'range': [3, 15],
        'description': '格拉斯哥昏迷评分'
    },
    'oasis': {
        'name': 'OASIS评分',
        'unit': '分',
        'range': [0, 100],
        'description': '牛津急性疾病严重程度评分'
    },
    'pH': {
        'name': '血液pH值',
        'unit': '',
        'range': [6.8, 7.8],
        'description': '动脉血pH值'
    }
}

@app.route('/')
def index():
    """返回主页"""
    return send_from_directory('static', 'index.html')

@app.route('/api/features', methods=['GET'])
def get_features():
    """获取特征信息"""
    return jsonify({
        'features': FEATURE_INFO,
        'feature_order': model.feature_name_ if model else []
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        if model is None:
            return jsonify({'error': '模型未加载'}), 500
        
        # 获取输入数据
        data = request.json
        
        # 验证输入
        if not data:
            return jsonify({'error': '未提供输入数据'}), 400
        
        # 按照模型训练时的特征顺序构建输入
        feature_values = []
        missing_features = []
        
        for feature_name in model.feature_name_:
            if feature_name not in data:
                missing_features.append(feature_name)
            else:
                try:
                    value = float(data[feature_name])
                    feature_values.append(value)
                except (ValueError, TypeError):
                    return jsonify({'error': f'特征 {feature_name} 的值无效'}), 400
        
        if missing_features:
            return jsonify({'error': f'缺少特征: {", ".join(missing_features)}'}), 400
        
        # 构建输入数组
        X = np.array([feature_values])
        
        # 进行预测
        # 预测概率
        prob = model.predict_proba(X)[0]
        risk_prob = float(prob[1])  # 高风险概率
        
        # 预测类别
        prediction = int(model.predict(X)[0])
        
        # 风险等级分类
        if risk_prob < 0.3:
            risk_level = '低风险'
            risk_color = 'green'
        elif risk_prob < 0.7:
            risk_level = '中风险'
            risk_color = 'orange'
        else:
            risk_level = '高风险'
            risk_color = 'red'
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'risk_probability': round(risk_prob * 100, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'input_features': {name: val for name, val in zip(model.feature_name_, feature_values)}
        })
        
    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测接口"""
    try:
        if model is None:
            return jsonify({'error': '模型未加载'}), 500
        
        data = request.json
        
        if not data or 'samples' not in data:
            return jsonify({'error': '未提供样本数据'}), 400
        
        samples = data['samples']
        results = []
        
        for idx, sample in enumerate(samples):
            feature_values = []
            for feature_name in model.feature_name_:
                if feature_name not in sample:
                    return jsonify({'error': f'样本 {idx+1} 缺少特征: {feature_name}'}), 400
                feature_values.append(float(sample[feature_name]))
            
            X = np.array([feature_values])
            prob = model.predict_proba(X)[0]
            risk_prob = float(prob[1])
            prediction = int(model.predict(X)[0])
            
            if risk_prob < 0.3:
                risk_level = '低风险'
            elif risk_prob < 0.7:
                risk_level = '中风险'
            else:
                risk_level = '高风险'
            
            results.append({
                'sample_id': idx + 1,
                'prediction': prediction,
                'risk_probability': round(risk_prob * 100, 2),
                'risk_level': risk_level
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_samples': len(samples)
        })
        
    except Exception as e:
        return jsonify({'error': f'批量预测失败: {str(e)}'}), 500

if __name__ == '__main__':
    # 加载模型
    if load_model():
        print("\n" + "="*50)
        print("LightGBM风险预测模型服务已启动")
        print("访问地址: http://localhost:5000")
        print("="*50 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("模型加载失败，无法启动服务")
