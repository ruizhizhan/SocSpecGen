import os
import re
from util.tools import fix_socrates_nan

def replace_negatives_in_cia_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.cia'):
            path = os.path.join(directory, filename)
            with open(path, 'r') as file:
                lines = file.readlines()

            # 检查并替换负数
            modified = False  # 标记文件是否被修改
            for i in range(len(lines)):  # 遍历所有行，不再硬编码跳过第一行
                parts = lines[i].split()
                if not parts: 
                    continue  # 跳过完全空白的行
                
                try:
                    # 尝试将这一行所有元素转为浮点数
                    numbers = list(map(float, parts))
                except ValueError:
                    # 包含 "CO2-CO2" 等字符串的行会引发错误，我们直接跳过这一行
                    continue

                # 替换负数为0
                replaced_numbers = [0 if num < 0 else num for num in numbers]
                if numbers != replaced_numbers:
                    modified = True
                    lines[i] = ' '.join(map(str, replaced_numbers)) + '\n'

            # 如果文件被修改，重新写入
            if modified:
                with open(path, 'w') as file:
                    file.writelines(lines)
                print(f"文件 {filename} 已被修改，所有负数已替换为0。")
                

def find_min_max_in_cia_files(directory):
    """
    遍历指定目录下的所有.cia文件，找出所有数字中的最大值和最小值。
    """
    min_value = float('inf')  # 初始化最小值为无穷大
    max_value = float('-inf')  # 初始化最大值为无穷小

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.cia'):
            path = os.path.join(directory, filename)
            
            # 在这里我改成了更节省内存的逐行读取方式，避免大文件占用过多内存
            with open(path, 'r') as file:
                for line in file:
                    parts = line.split()
                    if not parts:
                        continue
                        
                    try:
                        # 尝试将整行转换为浮点数
                        numbers = list(map(float, parts))
                    except ValueError:
                        # 遇到无法转换的信息行，直接跳过
                        continue
                    
                    if numbers: # 确保 numbers 列表不为空
                        # 更新最大值和最小值
                        current_min = min(numbers)
                        current_max = max(numbers)
                        min_value = min(min_value, current_min)
                        max_value = max(max_value, current_max)

    return min_value, max_value

def parse_spectral_files(directory_path, file_name):
    # 拼接完整的文件路径
    file_path = os.path.join(directory_path, f"{file_name}")
    file_k_path = os.path.join(directory_path, f"{file_name}_k")
    
    # 定义正则表达式
    # 匹配 "*BLOCK: TYPE = " 开头，提取后面的数字（支持整数和小数）
    block_pattern = re.compile(r'^\*BLOCK:\s*TYPE\s*=\s*([0-9\.]+)')
    # 匹配忽略大小写的单独词汇 "nan" (如 NaN, nan, NAN)。
    # 使用 \b 代表单词边界，防止匹配到 banana 里面的 nan
    nan_pattern = re.compile(r'\bnan\b', re.IGNORECASE) 
    
    block_numbers = []
    file_has_nan = False
    
    print(f"正在读取主文件: {file_path}")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 1. 查找 BLOCK: TYPE 后面的数字
                match = block_pattern.search(line)
                if match:
                    block_numbers.append(match.group(1))
                
                # 2. 查找是否有 nan
                # 如果你想匹配包含在其他字母内的 nan，可以将条件换成 'nan' in line.lower()
                if not file_has_nan and nan_pattern.search(line):
                    file_has_nan = True
    else:
        print(f"警告：未找到文件 {file_path}")

    # 读取 _k 文件 (虽然需求没有详细说明如何处理它，但在这里读取并检查 nan)
    file_k_has_nan = False
    print(f"正在读取附属文件: {file_k_path}")
    if os.path.exists(file_k_path):
        with open(file_k_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not file_k_has_nan and nan_pattern.search(line):
                    file_k_has_nan = True
    else:
        print(f"警告：未找到文件 {file_k_path}")

    # 输出结果
    print("-" * 30)
    print("分析结果:")
    print(f"文件 {file_name} 中 '*BLOCK: TYPE = ' 后面的数字有: {block_numbers}")
    print(f"文件 {file_name} 中是否包含 'nan': {'是' if file_has_nan else '否'}")
    print(f"文件 {file_name}_k 中是否包含 'nan': {'是' if file_k_has_nan else '否'}")
    
    return block_numbers, file_has_nan, file_k_has_nan

if __name__ == "__main__":
    # 1. gothrough the CIA files
    cia_root = '/work/home/ac9b0k6rio/SocSpecGen/hitran'
    for cia_rel_path in ['CO2-CO2_2024','CO2-H2O_2024','N2-H2O_2018','O2-CO2_2024','O2-N2_2024','N2-N2_2021','O2-O2_2024']:
        directory = os.path.join(cia_root, cia_rel_path)
        replace_negatives_in_cia_files(directory)
        min_value, max_value = find_min_max_in_cia_files(directory)
        print(f"所有文件中的最小值为: {min_value}")
        print(f"所有文件中的最大值为: {max_value}")
    
    # 2. check the blocks in spectral files and NaN
    spectral_file_dir = "/work/home/ac9b0k6rio/SocSpecGen/spectral_files/"
    for spectral_file_name in ["sp_b94/sp_sw_b94_Trappist-1_sphinx_SiO_T62xP22_001",
                               "sp_b96/sp_lw_b96_Trappist-1_sphinx_SiO_T62xP22_001",
                               "sp_b96/sp_lw_b96_55CancriA_CO2_T62xP22_001_nk20"]:
        parse_spectral_files(spectral_file_dir, spectral_file_name)
    
    # 3. fix NaN in socrates files
    # fix_socrates_nan(os.path.join(spectral_file_dir,"sp_b96/sp_lw_b96_Trappist-1_sphinx_SiO_T62xP22_001_k"))