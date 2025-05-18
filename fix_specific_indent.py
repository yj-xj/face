import os

def fix_specific_indentation():
    """专门修复face_swap.py文件中第197-202行的缩进问题"""
    file_path = "src/face_swap.py"
    backup_path = "src/face_swap.py.specific_fix.bak"
    
    print("开始修复face_swap.py中的特定缩进问题...")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在!")
        return False
    
    # 创建备份
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    with open(backup_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    print(f"已创建备份文件: {backup_path}")
    
    # 查找并修复特定的问题区域
    fixed = False
    indentation_base = "        "  # 基础缩进级别
    
    for i in range(len(lines)):
        # 找到 "if self.root is not None:" 的行
        if "if self.root is not None:" in lines[i] and i < 250:  # 只检查前250行
            print(f"找到if语句，行号: {i+1}")
            # 判断接下来的行应该是否需要缩进修复
            j = i + 1
            while j < len(lines) and j < i + 10:  # 检查接下来的几行
                # 如果发现缩进不正确的行
                if "self.root.title" in lines[j] or "self.root.geometry" in lines[j]:
                    # 检查缩进是否正确
                    if not lines[j].startswith(indentation_base + indentation_base):
                        lines[j] = indentation_base + indentation_base + lines[j].lstrip()
                        print(f"修复行 {j+1}: {lines[j].strip()}")
                        fixed = True
                elif "self.set_app_appearance" in lines[j]:
                    # 检查缩进是否正确
                    if not lines[j].startswith(indentation_base + indentation_base):
                        lines[j] = indentation_base + indentation_base + lines[j].lstrip()
                        print(f"修复行 {j+1}: {lines[j].strip()}")
                        fixed = True
                j += 1
    
    if fixed:
        # 保存修复后的文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)
        print(f"缩进问题已修复: {file_path}")
        return True
    else:
        print("未发现需要修复的缩进问题")
        return False

if __name__ == "__main__":
    success = fix_specific_indentation()
    print("修复程序执行完成" + (" - 成功!" if success else " - 未找到需要修复的问题!")) 