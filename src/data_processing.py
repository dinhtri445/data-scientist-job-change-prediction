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

#=====================Feature Engineering=====================
def min_max_scaling(col):
    """
    Đưa dữ liệu về khoảng [0, 1]
    Công thức: (X - min) / (max - min)
    """
    # Lưu ý: col ở đây phải là X_train hoặc X_test
    min_val = np.min(col)
    max_val = np.max(col)
    
    if max_val - min_val == 0: return col # Tránh chia cho 0
    
    return (col - min_val) / (max_val - min_val)

def log_transformation(col):
    """
    Giảm độ lệch của dữ liệu.
    Sử dụng log1p (log(1+x)) để tránh lỗi log(0) = -vocung
    """
    return np.log1p(col)

def decimal_scaling(col):
    """
    Công thức: x / 10^j (với j là số chữ số của giá trị lớn nhất)
    Ví dụ: 10005 -> chia cho 10^5 -> 0.10005
    """
    max_val = np.max(np.abs(col))
    if max_val == 0: return col
    
    # Tìm số mũ j (số chữ số)
    j = np.ceil(np.log10(max_val))
    
    return col / (10**j)

def z_score_standardization(col):
    """
    Công thức: (X - mean) / std
    """
    mean = np.mean(col)
    std = np.std(col)
    
    if std == 0: return np.zeros_like(col)
    
    return (col - mean) / std

#PCA