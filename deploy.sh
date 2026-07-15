#!/bin/bash

echo "=========================================="
echo "  检修助手 - 龙芯架构部署脚本"
echo "  LoongArch + 银河麒麟高级服务器版"
echo "=========================================="

WORK_DIR="/opt/repair-assistant"

if [ "$(id -u)" != "0" ]; then
    echo "错误：需要以 root 权限运行此脚本"
    echo "请使用: sudo bash deploy.sh"
    exit 1
fi

echo ""
echo "1. 创建项目目录..."
mkdir -p $WORK_DIR
chmod 755 $WORK_DIR

echo ""
echo "2. 安装系统依赖..."
apt update && apt install -y \
    python3 python3-pip python3-venv \
    gcc g++ make cmake \
    libopenblas-dev liblapack-dev \
    libjpeg-dev zlib1g-dev \
    poppler-utils \
    nginx \
    && echo "系统依赖安装完成"

echo ""
echo "3. 创建虚拟环境..."
python3 -m venv $WORK_DIR/venv
source $WORK_DIR/venv/bin/activate

echo ""
echo "4. 安装 Python 依赖..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "5. 创建数据目录..."
mkdir -p $WORK_DIR/uploads
mkdir -p $WORK_DIR/chroma_db
chown -R www-data:www-data $WORK_DIR/uploads
chown -R www-data:www-data $WORK_DIR/chroma_db

echo ""
echo "6. 创建 .env 文件..."
cat > $WORK_DIR/.env <<EOF
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MULTIMODAL_MODEL=qwen3.5-omni-plus
TEXT_MODEL=qwen-plus-0112
EOF
echo "请编辑 $WORK_DIR/.env 文件配置 API Key"

echo ""
echo "7. 创建 systemd 服务..."
cat > /etc/systemd/system/repair-assistant.service <<EOF
[Unit]
Description=检修助手服务
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=$WORK_DIR
Environment="PATH=$WORK_DIR/venv/bin"
ExecStart=$WORK_DIR/venv/bin/python start_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "8. 配置 Nginx 反向代理..."
cat > /etc/nginx/sites-available/repair-assistant <<EOF
server {
    listen 80;
    server_name localhost;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /static/ {
        alias $WORK_DIR/static/;
        expires 30d;
    }
}
EOF

ln -sf /etc/nginx/sites-available/repair-assistant /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

echo ""
echo "9. 启动服务..."
systemctl daemon-reload
systemctl enable repair-assistant
systemctl start repair-assistant

echo ""
echo "=========================================="
echo "  部署完成！"
echo ""
echo "  服务地址: http://服务器IP"
echo "  管理命令:"
echo "    查看状态: systemctl status repair-assistant"
echo "    重启服务: systemctl restart repair-assistant"
echo "    查看日志: journalctl -u repair-assistant -f"
echo ""
echo "  请务必编辑 $WORK_DIR/.env 配置 API Key"
echo "=========================================="