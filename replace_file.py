def replace_file():
    try:
        # 读取修复后的文件
        with open('src/face_swap_fixed_all.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 写入原始文件
        with open('src/face_swap.py', 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("文件替换成功！")
        return True
    except Exception as e:
        print(f"文件替换失败: {e}")
        return False

if __name__ == "__main__":
    replace_file() 