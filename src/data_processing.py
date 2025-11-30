import numpy as np
import math
#=====================Đọc Ghi File=====================
# Lấy header của file
def read_header(file_path):
    header = np.genfromtxt(file_path,
                        delimiter=",",
                        max_rows=1,
                        dtype=str,
                        encoding="utf-8")
    return header

# Lấy toàn bộ dữ liệu của file, kết quả lưu vào ma trận 2 chiều
def read_data(file_path):
    data = np.genfromtxt(file_path,
                    delimiter=",",
                    skip_header=1,
                    dtype = str,
                    encoding="utf-8")
    
    # Thay thế toàn bộ chuỗi '' trong mảng bằng chuỗi 'nan'
    data[data == ''] = 'nan'

    return data

# Lưu phải sau khi xử lý
def save_to_csv(data, header, file_path):
    try:
        header_str = ",".join(header)
        
        np.savetxt(file_path, 
                   data, 
                   delimiter=",",      
                   header=header_str, 
                   comments='',        
                   fmt='%s',       
                   encoding='utf-8')
                   
        print(f"✅ Đã lưu file thành công tại: {file_path}")
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")

#==================Thông tin cơ bản của dataset=======================
def _missing_mask(col):
    """Trả về mask True cho các giá trị missing ('' / 'nan' / 'None' / None)"""
    col = np.asarray(col)
    mask = np.zeros(col.shape, dtype=bool)
    for i, v in enumerate(col):
        if v is None:
            mask[i] = True
            continue
        s = v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else str(v)
        s = s.strip()
        if s == "" or s.lower() in ("nan", "none"):
            mask[i] = True
    return mask

def _detect_numeric(col):
    """Thử cast sang float. Nếu được trả về (arr_float, dtype_str),
       arr_float là float array có np.nan ở vị trí missing; dtype_str in {'int64','float64','object'}"""
    col = np.asarray(col)
    missing = _missing_mask(col)
    nonmiss = col[~missing]
    if nonmiss.size == 0:
        return None, 'object'
    try:
        floats = np.array([float(x) for x in nonmiss])
    except:
        return None, 'object'
    
    arr = np.full(col.shape, np.nan, dtype=float)
    arr[~missing] = floats
    # kiểm tra toàn là integer
    if np.all(np.floor(floats) == floats):
        return arr, 'int64'
    else:
        return arr, 'float64'

def np_info(data, header=None):
    """
    Hiển thị thông tin cơ bản giống df.info() (gọn).
    - data: structured array (dtype.names) hoặc 2D ndarray (n_rows, n_cols)
    - header: list tên cột nếu data là 2D ndarray
    """
    # chuẩn hoá input
    if hasattr(data, 'dtype') and data.dtype.names is not None:
        col_names = list(data.dtype.names)
        columns = [data[name] for name in col_names]
        n_rows = data.shape[0]
    else:
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError("data phải là structured array hoặc 2D ndarray")
        n_rows, n_cols = arr.shape
        col_names = list(header) if header is not None else [f"col_{i}" for i in range(n_cols)]
        columns = [arr[:, i] for i in range(n_cols)]

    print(f"Rows: {n_rows}, Columns: {len(col_names)}")
    print("-" * 48)
    print(" #  Column".ljust(28) + "Non-Null  Dtype")
    print("--- " + "-"*24 + "  " + "-"*5)
    for i, name in enumerate(col_names):
        col = columns[i]
        missing = _missing_mask(col)
        non_null = int(np.sum(~missing))
        _, dtype = _detect_numeric(col)
        print(f"{i:<3} {name.ljust(24)} {str(non_null).ljust(8)} {dtype}")
    print("-" * 48)

def np_describe(data, header=None, float_digits=6):
    """
    In thống kê cho cột numeric, hiển thị gọn
    """
    # chuẩn hoá input -> danh sách cột và tên
    if hasattr(data, 'dtype') and data.dtype.names is not None:
        col_names = list(data.dtype.names)
        cols = [data[n] for n in col_names]
    else:
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError("data phải là structured array hoặc 2D ndarray")
        nrows, ncols = arr.shape
        col_names = list(header) if header is not None else [f"col_{i}" for i in range(ncols)]
        cols = [arr[:, i] for i in range(ncols)]

    # thu các cột numeric
    numeric_names = []
    numeric_arrays = []
    for n, c in zip(col_names, cols):
        arr_float, dtype = _detect_numeric(c)
        if arr_float is not None:
            numeric_names.append(n)
            numeric_arrays.append(arr_float)

    if len(numeric_names) == 0:
        print("Không có cột số để mô tả.")
        return

    # chuẩn bị dữ liệu chuỗi cho mỗi stat x col
    stats = ['count','mean','std','min','25%','50%','75%','max']
    table = {s: [] for s in stats}

    for arr in numeric_arrays:
        clean = arr[~np.isnan(arr)]
        if clean.size == 0:
            vals = [0.0, float('nan'), float('nan'), float('nan'),
                    float('nan'), float('nan'), float('nan'), float('nan')]
        else:
            vals = [
                float(clean.size),
                float(np.mean(clean)),
                float(np.std(clean, ddof=1)) if clean.size>1 else 0.0,
                float(np.min(clean)),
                float(np.percentile(clean,25)),
                float(np.median(clean)),
                float(np.percentile(clean,75)),
                float(np.max(clean))
            ]
        for s, v in zip(stats, vals):
            table[s].append(v)

    # Định dạng số thành chuỗi
    def fmt_val(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "NaN"
        
        if isinstance(v, float) and v.is_integer():
            return f"{int(v)}"
    
        s = f"{v:.{float_digits}f}"
        # bỏ các số 0 và dấu chấm ở cuối
        s = s.rstrip('0').rstrip('.')
        if s == "-0":
            s = "0"
        return s

    # xây dựng bảng chuỗi
    str_table = {s: [fmt_val(v) for v in table[s]] for s in stats}
    
    col_widths = []
    for j, name in enumerate(numeric_names):
        max_len = len(name)
        for s in stats:
            max_len = max(max_len, len(str_table[s][j]))
        # thêm phần đệm nhỏ
        col_widths.append(max_len + 2)

    # in hàng tiêu đề
    first_col_width = 12
    header_row = " " * first_col_width
    for name, w in zip(numeric_names, col_widths):
        header_row += name.center(w)
    print(header_row)

    # in từng hàng số liệu thống kê
    for s in stats:
        row = s.ljust(first_col_width)
        for j, w in enumerate(col_widths):
            val = str_table[s][j]
            row += val.rjust(w)
        print(row)
#=====================Convert=====================

def convert_experience(column):
    col = column.copy()

    col[col == '>20'] = '21'
    col[col == '<1'] = '0'

    return col.astype(float)
 

def convert_company_size(column):
    col = column.copy()

    col[col == '50-99'] = '75'
    col[col == '<10'] = '9'
    col[col == '10000+'] = '10005'
    col[col == '5000-9999'] = '7500'
    col[col == '1000-4999'] = '3000'
    col[col == '10/49'] = '10'
    col[col == '100-500'] = '300'
    col[col == '500-999'] = '750'
    
    return col.astype(float)
    
def convert_last_new_job(column):
    col = column.copy()

    col[col == '>4'] = '5'
    col[col == 'never'] = '0'
   
    return col.astype(float)

#==========================================
# Xử lý giá trị ngoại lai
def handle_outliers_iqr(col):
  
    col_float = col.astype(float)

    Q1 = np.nanpercentile(col_float, 25)
    Q3 = np.nanpercentile(col_float, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    col_clipped = np.clip(col_float, lower_bound, upper_bound)
    
    return col_clipped.astype(str) 

#=====================Feature Engineering=====================
def create_enrollment_status(enrolled_col):
    status = np.ones(enrolled_col.shape, dtype=int)

    mask = (enrolled_col == 'Unknown') | (enrolled_col == 'no_enrollment')
    status[mask] = 0

    return status

def create_is_stem(major_col):
    # Tạo mảng số 0
    is_stem = np.zeros(major_col.shape, dtype=int)
    # Chỗ nào là STEM thì gán 1
    is_stem[major_col == 'STEM'] = 1

    return is_stem

def create_experience_level(exp_col_float):
    # Tạo mảng kết quả
    levels = np.zeros(exp_col_float.shape, dtype=int)
    
    # Gán giá trị theo điều kiện
    levels[(exp_col_float >= 3) & (exp_col_float <= 7)] = 1  # Mid
    levels[(exp_col_float >= 8) & (exp_col_float <= 15)] = 2 # Senior
    levels[exp_col_float > 15] = 3                           # Expert
    
    return levels

def create_stability_ratio(last_job_col, exp_col):
    """
    Tạo chỉ số ổn định = last_new_job / experience
    """
    numerator = last_job_col.astype(float) 
    denominator = exp_col.astype(float)  
    
    ratio = np.zeros(numerator.shape, dtype=float)
    
    mask = denominator > 0
    # Thực hiện chia (Vector hóa)
    ratio[mask] = numerator[mask] / denominator[mask]
    
    return np.round(ratio, 2)

#=====================Kiểm định giả thuyết thống kê=====================
# Hàm tính Chi-Square (Dành cho biến Phân loại - Categorical)
def calculate_chi_square_test(feature_col, target_col):
    """
    Tính thống kê Chi-bình phương (Chi-Square Statistic) để kiểm định tính độc lập.
    """
    # Tạo bảng chéo thủ công bằng NumPy
    categories = np.unique(feature_col)
    target_groups = np.unique(target_col)
    
    observed = np.array([
        [np.sum((feature_col == cat) & (target_col == tgt)) for tgt in target_groups]
        for cat in categories
    ])
    
    # Tính các tổng (Row sums, Col sums, Total)
    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)
    total = observed.sum()
    
    # Tính bảng tần số kỳ vọng (Expected Frequencies)
    # Dùng phép nhân ngoài (Outer product) để tạo ma trận nhanh chóng
    expected = np.outer(row_sums, col_sums) / total
    
    # Tránh lỗi chia cho 0
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2_components = (observed - expected)**2 / expected
        chi2_components[~np.isfinite(chi2_components)] = 0 # Thay thế nan/inf bằng 0
        
    chi2_score = chi2_components.sum()
    
    # Tính bậc tự do 
    dof = (len(categories) - 1) * (len(target_groups) - 1)
    
    return chi2_score, dof

# Hàm tính T-Test (Dành cho biến Số - Numerical)
def calculate_t_test(feature_col, target_col):
    # Chuyển đổi dữ liệu sang float để tính toán 
    try:
        feature_float = feature_col.astype(float)
        target_float = target_col.astype(float)
    except ValueError:
        print("Lỗi: Dữ liệu đầu vào không phải là số hợp lệ.")
        return 0.0

    # Tách dữ liệu thành 2 nhóm dựa trên Target (0 và 1)
    group0 = feature_float[target_float == 0]
    group1 = feature_float[target_float == 1]
    
    # Tính các tham số thống kê cơ bản (Mean, Variance, N)
    n0, n1 = len(group0), len(group1)
    
    # Nếu một trong hai nhóm rỗng, không thể tính toán
    if n0 == 0 or n1 == 0:
        return 0.0
        
    mean0, mean1 = np.mean(group0), np.mean(group1)
    
    var0, var1 = np.var(group0, ddof=1), np.var(group1, ddof=1)
    
    # ính Sai số chuẩn (Standard Error - SE)
    # Công thức: sqrt( s1^2/n1 + s2^2/n2 )
    se = np.sqrt((var0/n0) + (var1/n1))
    
    # Tính T-score
    # Tránh lỗi chia cho 0
    if se == 0:
        return 0.0
        
    t_stat = (mean0 - mean1) / se
    
    return t_stat
#=====================Modeling=====================
# Hàm chia tập train/test
def train_test_split_numpy(X, y, test_size=0.2, seed=None):
    """
    Chia dữ liệu thành tập Train và Test ngẫu nhiên.
    """
    if seed:
        np.random.seed(seed)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_count = int(n_samples * test_size)
    train_count = n_samples - test_count
    
    # Phân chia index
    train_idx = indices[:train_count]
    test_idx = indices[train_count:]
    
    # Cắt dữ liệu
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test

# Hàm mã hóa (Encoding)
def label_encode_numpy(column):
    """
    Chuyển đổi cột chữ thành số (0, 1, 2...).
    """

    uniques = np.unique(column)
   
    mapping = {val: i for i, val in enumerate(uniques)}
    
    encoded_col = np.array([mapping[val] for val in column], dtype=int)
    
    return encoded_col, mapping

# Hàm chuẩn hóa Z-SCORE (Standardization)
def standard_scaler_numpy(X_train, X_test):
    """
    Đưa dữ liệu về phân phối chuẩn (Mean=0, Std=1).
    Quan trọng: Tính Mean/Std trên Train, áp dụng cho cả Train và Test.
    """
    # Tính tham số trên tập TRAIN
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Tránh chia cho 0 
    std[std == 0] = 1.0 
    
    # Áp dụng công thức: (X - mean) / std
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled

# Hàm chuẩn hóa MIN-MAX (Normalization)
def min_max_scaler_numpy(X_train, X_test):
    """
    Đưa dữ liệu về khoảng [0, 1].
    """
    # Tính tham số trên tập TRAIN
    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)
    
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0 # Tránh chia cho 0
    
    # Công thức: (X - min) / (max - min)
    X_train_scaled = (X_train - min_val) / range_val
    X_test_scaled = (X_test - min_val) / range_val
    
    return X_train_scaled, X_test_scaled

# Hàm giảm chiều dữ liệu
def pca_numpy(X_train, X_test, n_components=None):
    """
    Thực hiện PCA để giảm chiều dữ liệu.
    Nguyên tắc: Fit trên Train, Transform trên cả Train và Test.
    """
    # Tính ma trận hiệp phương sai (Covariance Matrix) của TRAIN
    n_samples = X_train.shape[0]
    covariance_matrix = np.dot(X_train.T, X_train) / (n_samples - 1)
    
    # Phân rã trị riêng, vector riêng (Eigen decomposition)
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
    
    # Sắp xếp Vector riêng theo thứ tự giảm dần của Trị riêng
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    sorted_eigenvalues = eigen_values[sorted_index]
    
    # Chọn k thành phần (n_components)
    if n_components is None:
        # Nếu không chọn, giữ lại số thành phần giải thích 95% phương sai
        total_var = np.sum(sorted_eigenvalues)
        explained_variance_ratio = sorted_eigenvalues / total_var
        cumulative_variance = np.cumsum(explained_variance_ratio)
        n_components = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"PCA tự động chọn {n_components} thành phần (giữ lại 95% thông tin).")
        
    eigenvector_subset = sorted_eigenvectors[:, 0:n_components]
    
    # Chiếu dữ liệu lên không gian mới (Transform)
    X_train_pca = np.dot(X_train, eigenvector_subset)
    X_test_pca = np.dot(X_test, eigenvector_subset)
    
    return X_train_pca, X_test_pca

# Dùng SMOTE để cân bằng dữ liệu
def smote_numpy(X, y, k_neighbors=5, random_state=42):
    """
    Thực hiện SMOTE (Synthetic Minority Over-sampling Technique) dùng NumPy.
    """
    np.random.seed(random_state)
    
    # Tách nhóm thiểu số (Class 1) và đa số (Class 0)
    X_minority = X[y == 1]
    X_majority = X[y == 0]
    
    n_minority = len(X_minority)
    n_majority = len(X_majority)
    
    # Tính số lượng mẫu cần sinh thêm (để cân bằng 50-50)
    n_synthetic = n_majority - n_minority
    
    if n_synthetic <= 0:
        return X, y # Không cần SMOTE nếu Class 1 đã nhiều hơn
    
    print(f"SMOTE: Đang sinh thêm {n_synthetic} mẫu giả lập...")
    
    synthetic_samples = []
    
    # Vòng lặp sinh dữ liệu
    # Duyệt qua các mẫu gốc để sinh mẫu mới
    # Tính số lượng mẫu mới cần sinh ra từ MỖI mẫu cũ
    for _ in range(n_synthetic):
        # Chọn ngẫu nhiên 1 điểm gốc (Parent) từ nhóm thiểu số
        idx = np.random.randint(0, n_minority)
        sample = X_minority[idx]
        
        # Tìm k hàng xóm gần nhất (KNN logic)
        distances = np.sqrt(np.sum((X_minority - sample)**2, axis=1))
        
        # Lấy index của k điểm gần nhất
        neighbor_indices = np.argsort(distances)[1:k_neighbors+1]
        
        # Chọn ngẫu nhiên 1 hàng xóm (Neighbor)
        random_neighbor_idx = np.random.choice(neighbor_indices)
        neighbor = X_minority[random_neighbor_idx]
        
        # Tạo điểm mới (Interpolation)
        # Công thức: New = Old + (Neighbor - Old) * random(0, 1)
        diff = neighbor - sample
        gap = np.random.rand() 
        new_sample = sample + (diff * gap)
        
        synthetic_samples.append(new_sample)
        
    # Gộp lại thành dữ liệu mới
    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.ones(len(X_synthetic))
    
    # Ghép với dữ liệu gốc (Bao gồm cả X_majority và X_minority cũ)
    X_final = np.vstack((X_majority, X_minority, X_synthetic))
    y_final = np.hstack((np.zeros(n_majority), np.ones(n_minority), y_synthetic))
    
    # Xáo trộn (Shuffle)
    shuffle_idx = np.arange(len(y_final))
    np.random.shuffle(shuffle_idx)
    
    print(f"Hoàn tất SMOTE. Kích thước mới: {X_final.shape}")
    return X_final[shuffle_idx], y_final[shuffle_idx]