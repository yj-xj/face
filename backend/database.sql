-- 人脸替换应用数据库结构
-- 创建数据库
CREATE DATABASE IF NOT EXISTS face_swap_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE face_swap_db;

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 人脸图片表
CREATE TABLE IF NOT EXISTS face_images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL COMMENT '文件大小(字节)',
    width INT COMMENT '图片宽度',
    height INT COMMENT '图片高度',
    thumbnail_path VARCHAR(500) COMMENT '缩略图路径',
    face_count INT DEFAULT 0 COMMENT '检测到的人脸数量',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_is_active (is_active),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 输入视频表
CREATE TABLE IF NOT EXISTS input_videos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL COMMENT '文件大小(字节)',
    duration FLOAT COMMENT '视频时长(秒)',
    width INT COMMENT '视频宽度',
    height INT COMMENT '视频高度',
    fps FLOAT COMMENT '帧率',
    codec VARCHAR(50) COMMENT '编码格式',
    thumbnail_path VARCHAR(500) COMMENT '缩略图路径',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_is_active (is_active),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 输出视频表
CREATE TABLE IF NOT EXISTS output_videos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    input_video_id INT,
    face_image_id INT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL COMMENT '文件大小(字节)',
    duration FLOAT COMMENT '视频时长(秒)',
    width INT COMMENT '视频宽度',
    height INT COMMENT '视频高度',
    fps FLOAT COMMENT '帧率',
    processing_method ENUM('traditional', 'inswapper') DEFAULT 'inswapper',
    processing_time FLOAT COMMENT '处理耗时(秒)',
    thumbnail_path VARCHAR(500) COMMENT '缩略图路径',
    status ENUM('processing', 'completed', 'failed') DEFAULT 'processing',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (input_video_id) REFERENCES input_videos(id) ON DELETE SET NULL,
    FOREIGN KEY (face_image_id) REFERENCES face_images(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 处理任务表
CREATE TABLE IF NOT EXISTS processing_tasks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    task_type ENUM('video', 'camera') NOT NULL,
    input_video_id INT,
    face_image_id INT NOT NULL,
    status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
    progress INT DEFAULT 0 COMMENT '进度 0-100',
    error_message TEXT,
    processing_params JSON COMMENT '处理参数',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (input_video_id) REFERENCES input_videos(id) ON DELETE SET NULL,
    FOREIGN KEY (face_image_id) REFERENCES face_images(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 系统配置表
CREATE TABLE IF NOT EXISTS system_config (
    id INT AUTO_INCREMENT PRIMARY KEY,
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value TEXT NOT NULL,
    description VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_config_key (config_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 插入默认配置
INSERT INTO system_config (config_key, config_value, description) VALUES
('max_upload_size', '524288000', '最大上传文件大小(字节) - 500MB'),
('allowed_video_formats', 'mp4,avi,mov,mkv,flv,wmv', '允许的视频格式'),
('allowed_image_formats', 'jpg,jpeg,png,bmp', '允许的图片格式'),
('storage_path', 'uploads/', '文件存储路径'),
('max_concurrent_tasks', '3', '最大并发处理任务数')
ON DUPLICATE KEY UPDATE config_value = VALUES(config_value);
