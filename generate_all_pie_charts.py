
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from visualization_app.utils import generate_clean_pie_chart

# 数据集配置
DATA_DIRS = {
    # "Visium": {
    #     "result_dir": "output/visium_results",
    #     "data_dirs": ["data/visium_combined"]
    # },
    "seqFISH+": {
        "result_dir": "output/seqfish_results",
        "data_dirs": ["data/seqfish_tsv"]
    },
    "STARmap": {
        "result_dir": "output/stdgcn_starmap",
        "data_dirs": ["data/starmap_tsv"]
    }
}

def load_data(result_dir, data_dirs_list):
    """加载预测结果和坐标"""
    predict_path = os.path.join(result_dir, "predict_result.csv")
    if not os.path.exists(predict_path):
        return None, None
    
    predict_df = pd.read_csv(predict_path, index_col=0)
    
    coords = None
    for data_dir in data_dirs_list:
        coord_path = os.path.join(data_dir, "coordinates.csv")
        if os.path.exists(coord_path):
            try:
                temp_coords = pd.read_csv(coord_path, index_col=0)
                if len(temp_coords) == len(predict_df):
                    coords = temp_coords
                    break
            except:
                continue
    return predict_df, coords

def main():
    print("🚀 开始批量生成饼图背景...")
    
    for name, paths in DATA_DIRS.items():
        print(f"\n[处理数据集: {name}]")
        result_dir = paths["result_dir"]
        data_dirs = paths["data_dirs"]
        
        # 1. 加载数据
        predict_df, coords = load_data(result_dir, data_dirs)
        
        if predict_df is None or coords is None:
            print(f"  ❌ 未找到完整数据，跳过。")
            continue
            
        print(f"  📊 加载成功：{len(predict_df)} 个点")
        
        # 2. 生成图片
        print("  🎨 正在绘制饼图 (可能需要一些时间)...")
        # 传入 None 以启用自动点大小计算
        img, (xlim, ylim) = generate_clean_pie_chart(predict_df, coords, point_size=None)
        
        # 3. 保存图片
        output_img_path = os.path.join(result_dir, "interactive_pie_background.png")
        img.save(output_img_path)
        print(f"  ✅ 图片已保存: {output_img_path}")
        
        # 4. 保存元数据 (坐标范围)，这对于 Plotly 对齐至关重要
        metadata = {
            "xlim": xlim,
            "ylim": ylim
        }
        output_meta_path = os.path.join(result_dir, "interactive_pie_bounds.json")
        with open(output_meta_path, 'w') as f:
            json.dump(metadata, f)
        print(f"  ✅ 元数据已保存: {output_meta_path}")

    print("\n🎉 所有任务完成！")

if __name__ == "__main__":
    main()
