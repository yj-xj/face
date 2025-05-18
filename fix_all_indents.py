import os
import re

def fix_all_indentation_issues():
    """修复face_swap.py文件中的所有缩进问题"""
    file_path = "src/face_swap.py"
    backup_path = "src/face_swap.py.all_fixed.bak"
    
    print("开始修复face_swap.py中的所有缩进问题...")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在!")
        return False
    
    # 创建备份
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 保存备份
    with open(backup_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"已创建备份文件: {backup_path}")
    
    # 修复方式1: 使用正则表达式修复特定模式
    fixed = False
    
    # 直接修复第310行附近的if self.root is not None:后面的show_model_download_guide
    pattern = r'(if self\.root is not None:\n)(\s*)self\.show_model_download_guide'
    if re.search(pattern, content):
        content = re.sub(pattern, r'\1            self.show_model_download_guide', content)
        print("修复了if self.root is not None:之后的self.show_model_download_guide的缩进")
        fixed = True
    
    # 保存修改
    if fixed:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"缩进问题已修复: {file_path}")
        return True
    else:
        # 尝试修复方式2: 逐行读取修复
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # 定位第310行附近的问题
        for i in range(300, 320):
            if i < len(lines):
                line = lines[i]
                # 打印行内容以便调试
                print(f"行 {i+1}: {line.strip()}")
                # 检查是否是需要修复的行
                if "if self.root is not None:" in line:
                    print(f"找到if语句，行号: {i+1}")
                    # 检查下一行
                    if i+1 < len(lines) and "self.show_model_download_guide" in lines[i+1]:
                        # 检查缩进
                        if not lines[i+1].startswith("            "):  # 12个空格
                            lines[i+1] = "            " + lines[i+1].lstrip()
                            print(f"修复行 {i+2}: {lines[i+1].strip()}")
                            fixed = True
        
        # 保存修改
        if fixed:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(lines)
            print(f"缩进问题已修复: {file_path}")
            return True
        else:
            print("未发现需要修复的缩进问题")
            return False

if __name__ == "__main__":
    success = fix_all_indentation_issues()
    print("修复程序执行完成" + (" - 成功!" if success else " - 未找到需要修复的问题!")) 