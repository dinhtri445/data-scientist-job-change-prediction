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

# Trực quan hóa cho câu hỏi 1
def plot_job_hopping_analysis(data, col_idx):
    """
    Vẽ biểu đồ phân tích mối quan hệ giữa Lịch sử nhảy việc (last_new_job) 
    và Tỷ lệ nghỉ việc (Target).
    """
    last_job_vals = data[:, col_idx['last_new_job']].astype(float)
    target_vals = data[:, col_idx['target']].astype(float)
    
    plt.figure(figsize=(10, 6))
    
    ax = sns.barplot(x=last_job_vals, y=target_vals, 
                     palette="magma", errorbar=None)

    # Vì dữ liệu là số 0, 1, 2, 3, 4, 5
    # Hiển thị thành chữ cho dễ hiểu
    new_labels = ['Never (0)', '1 Year', '2 Years', '3 Years', '4 Years', '>4 Years (5)']
    ax.set_xticklabels(new_labels)
    
    plt.title('Tỷ lệ Nghỉ việc dựa trên Lịch sử thay đổi công việc', fontsize=14, fontweight='bold')
    plt.xlabel('Khoảng cách kể từ lần đổi việc trước', fontsize=12)
    plt.ylabel('Tỷ lệ muốn nghỉ việc', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Hiển thị số liệu
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f', padding=3)
        
    plt.show()

# Trực quan hóa cho câu hỏi 2
def plot_learning_analysis(data, col_idx):
    """
    Phân tích yếu tố học tập
    """
    hours_vals = data[:, col_idx['training_hours']].astype(float)
    target_vals = data[:, col_idx['target']].astype(float)
    learner_vals = data[:, col_idx['is_active_learner']].astype(float).astype(int) # '0' hoặc '1'
    
    # Khởi tạo khung hình (1 hàng, 2 cột)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- BIỂU ĐỒ 1: Training Hours (Boxplot) ---
    sns.boxplot(x=target_vals, y=hours_vals, ax=axes[0], palette="Set2")
    axes[0].set_title('Phân bố Giờ đào tạo theo Quyết định nghỉ việc', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Quyết định (0: Ở lại, 1: Nghỉ)', fontsize=10)
    axes[0].set_ylabel('Số giờ đào tạo', fontsize=10)
    
    # --- BIỂU ĐỒ 2: Active Learner (Barplot) ---
    # Vẽ tỷ lệ nghỉ việc (Mean of Target)
    sns.barplot(x=learner_vals, y=target_vals, ax=axes[1], 
                palette="Set1", errorbar=None)
    
    axes[1].set_title('Tỷ lệ Nghỉ việc: Đang đi học (1) vs Không (0)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Trạng thái Active Learner', fontsize=10)
    axes[1].set_ylabel('Tỷ lệ nghỉ việc', fontsize=10)
    
    # Hiển thị số liệu trên cột
    for i in axes[1].containers:
        axes[1].bar_label(i, fmt='%.2f', padding=3)

    plt.tight_layout()
    plt.show()

# Trực quan hóa cho câu hỏi 3
def plot_city_experience_heatmap(data, col_idx):
    """
    Vẽ Heatmap để tìm 'Điểm nóng' giữa Mức độ phát triển thành phố và Kinh nghiệm.
    """
    city_vals = data[:, col_idx['city_development_index']].astype(float)
    exp_vals = data[:, col_idx['experience_level']].astype(float).astype(int)
    target_vals = data[:, col_idx['target']].astype(float)
    
    # Phân nhóm Thành phố (Binning) bằng NumPy
    # 0: Low (< 0.65), 1: Medium (0.65 - 0.85), 2: High (> 0.85)
    city_bins = np.zeros(city_vals.shape, dtype=int)
    city_bins[(city_vals >= 0.65) & (city_vals <= 0.85)] = 1 # Medium
    city_bins[city_vals > 0.85] = 2 # High
    
    # Nhãn cho trục
    city_labels = ['Low Dev (<0.65)', 'Medium Dev', 'High Dev (>0.85)']
    exp_labels = ['Junior', 'Mid-level', 'Senior', 'Expert']
    
    # Tạo Ma trận 2 chiều (4 hàng Exp x 3 cột City) để chứa tỷ lệ Target
    heatmap_matrix = np.zeros((4, 3))
    
    for i in range(4): 
        for j in range(3): 
            mask = (exp_vals == i) & (city_bins == j)
            
            if np.sum(mask) > 0:
                # Tính trung bình Target (Tỷ lệ nghỉ việc)
                avg_target = np.mean(target_vals[mask])
                heatmap_matrix[i, j] = avg_target
            else:
                heatmap_matrix[i, j] = 0 
                
    # Vẽ Heatmap
    plt.figure(figsize=(8, 6))
    
    
    sns.heatmap(heatmap_matrix, annot=True, fmt=".1%", cmap="Reds",
                xticklabels=city_labels, yticklabels=exp_labels)
    
    plt.title('Tỷ lệ Nghỉ việc: Kinh nghiệm vs Môi trường sống', fontsize=14, fontweight='bold')
    plt.xlabel('Mức độ phát triển Thành phố', fontsize=12)
    plt.ylabel('Trình độ Kinh nghiệm', fontsize=12)
    plt.show()

# Trực quan hóa cho câu hỏi 4
def plot_stem_analysis(data, col_idx):
    """
    Vẽ biểu đồ so sánh tỷ lệ nghỉ việc giữa Dân STEM và Dân Ngoại đạo (Non-STEM).
    """
    stem_vals = data[:, col_idx['is_stem_major']].astype(float).astype(int)
    target_vals = data[:, col_idx['target']].astype(float)
    
    plt.figure(figsize=(8, 5))
    
    # Vẽ Barplot (Tính trung bình tỷ lệ nghỉ việc)
    ax = sns.barplot(x=stem_vals, y=target_vals, 
                     palette="Blues", errorbar=None)
    

    plt.title('Tỷ lệ Nghỉ việc: Dân Ngoại đạo (0) vs Dân STEM (1)', fontsize=14, fontweight='bold')
    plt.xlabel('Nền tảng chuyên môn', fontsize=12)
    plt.ylabel('Tỷ lệ nghỉ việc (Trung bình)', fontsize=12)
    
    # Giới hạn trục Y để nhìn rõ sự chênh lệch (nếu có)
    plt.ylim(0, 0.4) 
    
    # Đổi nhãn trục X cho dễ hiểu
    ax.set_xticklabels(['Non-STEM (0)', 'STEM (1)'])

    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f', padding=3, fontsize=11, fontweight='bold')
        
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

# Trực quan hóa cho câu hỏi 5
def plot_overqualification_analysis(data, col_idx):
    """
    So sánh tỷ lệ nghỉ việc dựa trên Trình độ học vấn vs Quy mô công ty.
    """

    edu_vals = data[:, col_idx['education_level']]
    size_vals = data[:, col_idx['company_size']].astype(float)
    target_vals = data[:, col_idx['target']].astype(float)
    
    # Gom nhóm Company Size (Binning) để vẽ cho gọn
    # Small (<100), Medium (100-999), Large (>=1000)
    size_cats = np.empty(size_vals.shape, dtype=object)
    size_cats[size_vals < 100] = '1. Small (<100)'
    size_cats[(size_vals >= 100) & (size_vals < 1000)] = '2. Medium'
    size_cats[size_vals >= 1000] = '3. Large (1000+)'
    
    # Lọc dữ liệu: Chỉ quan tâm nhóm có bằng cấp 
    # Để tập trung vào câu hỏi: Graduate vs Masters vs PhD
    focus_degrees = ['Graduate', 'Masters', 'Phd']
    
    # Tạo ma trận kết quả (3 hàng Degree x 3 cột Size)
    heatmap_matrix = np.zeros((3, 3))
    size_labels = ['1. Small (<100)', '2. Medium', '3. Large (1000+)']
    
    for i, degree in enumerate(focus_degrees):
        for j, size_cat in enumerate(size_labels):
            # Lọc mặt nạ (Mask)
            mask = (edu_vals == degree) & (size_cats == size_cat)
            
            if np.sum(mask) > 0:
                heatmap_matrix[i, j] = np.mean(target_vals[mask])
            else:
                heatmap_matrix[i, j] = 0
                
    # 4. Vẽ Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_matrix, annot=True, fmt=".1%", cmap="Reds",
                xticklabels=size_labels, yticklabels=focus_degrees)
    
    plt.title('Tỷ lệ Nghỉ việc: Bằng cấp vs Quy mô Công ty', fontsize=14, fontweight='bold')
    plt.xlabel('Quy mô Công ty', fontsize=12)
    plt.ylabel('Trình độ Học vấn', fontsize=12)
    plt.show()