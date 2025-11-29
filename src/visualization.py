import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_missing_heatmap(data, col_names=None, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    
    data_str = data.astype(str)
    
    missing_mask = (data_str == 'nan')
    
    ax = sns.heatmap(missing_mask, 
                     cmap='viridis',    
                     vmin=0, vmax=1)    
    

    if col_names is not None:

        ax.set_xticks(np.arange(len(col_names)) + 0.5)
        ax.set_xticklabels(col_names, rotation=45, ha='right', fontsize=10)
       
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    
    plt.title("Missing Values Map", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

# # Biểu đồ hộp (Boxplot) để phát hiện giá trị ngoại lai.
# def plot_boxplot(data, col_idx, col_name):
   
#     idx = col_idx.get(col_name)
#     if idx is None:
#         print(f"Lỗi: Không tìm thấy cột '{col_name}'")
#         return
        
#     col_data = data[:, idx]
#     try:
#         col_float = col_data.copy()
        
#         col_float = col_float.astype(float)
        
#         # Loại bỏ nan để vẽ cho đẹp 
#         col_clean = col_float[~np.isnan(col_float)]
        
#     except Exception as e:
#         print(f"Lỗi: Không thể chuyển cột '{col_name}' sang số để vẽ. Kiểm tra lại dữ liệu.")
#         print(e)
#         return

#     plt.figure(figsize=(10, 4))
    
#     # Vẽ Boxplot nằm ngang 
#     sns.boxplot(x=col_clean, color='skyblue')
    
#     plt.title(f"Phân phối và Ngoại lai của cột: {col_name}", fontsize=14)
#     plt.xlabel("Giá trị", fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.show()
    



