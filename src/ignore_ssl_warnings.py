import os
import urllib3
import warnings
import ssl
import certifi
import requests

# 禁用SSL验证
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 设置环境变量
os.environ['PYTHONWARNINGS'] = 'ignore:Unverified HTTPS request'
os.environ['INSIGHTFACE_ALLOW_LOCAL_MODEL'] = '1'

# 修补requests库
old_merge_environment_settings = requests.Session.merge_environment_settings

def new_merge_environment_settings(self, url, proxies, stream, verify, cert):
    settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
    settings['verify'] = False
    return settings

requests.Session.merge_environment_settings = new_merge_environment_settings

def patch_insightface_download():
    """修补insightface的下载功能，禁用SSL验证"""
    try:
        import insightface.utils.storage
        import insightface.utils.download
        
        # 尝试修补新版本download.py
        try:
            # 对于0.7版本以上的insightface
            if hasattr(insightface.utils.download, 'download_file'):
                old_download = insightface.utils.download.download_file
                
                def new_download(url, path=None, overwrite=False, sha1_hash=None, retries=5, verify=False):
                    return old_download(url, path, overwrite, sha1_hash, retries, verify=False)
                
                insightface.utils.download.download_file = new_download
                print("成功修补insightface.utils.download.download_file函数")
        except Exception as e:
            print(f"修补download_file函数时出错: {e}")
        
        # 尝试修补storage.py中的download函数
        try:
            if hasattr(insightface.utils.storage, 'download'):
                old_storage_download = insightface.utils.storage.download
                
                def new_storage_download(sub_dir, name, force=False, root=None):
                    # 设置环境变量以允许使用本地模型
                    os.environ['INSIGHTFACE_ALLOW_LOCAL_MODEL'] = '1'
                    if root is None:
                        root = os.path.expanduser(os.path.join('~', '.insightface'))
                    
                    # 检查模型是否已经存在
                    final_dir = os.path.join(root, sub_dir, name)
                    if os.path.exists(final_dir) and not force:
                        return final_dir
                    
                    return old_storage_download(sub_dir, name, force, root)
                
                insightface.utils.storage.download = new_storage_download
                print("成功修补insightface.utils.storage.download函数")
        except Exception as e:
            print(f"修补storage.download函数时出错: {e}")
            
    except ImportError:
        print("未找到insightface模块，跳过修补")
    except Exception as e:
        print(f"修补insightface下载功能时出错: {e}")

# 执行修补
patch_insightface_download() 