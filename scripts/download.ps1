# 安装git lfs
git lfs install

# 下载数据集
git clone https://huggingface.co/datasets/DY-Evalab/EvalMuse temp_evalmuse

# 进入目录
cd temp_evalmuse

# 合并分片文件
Get-Content images.zip.part-* | Set-Content -Path images.zip

# 解压到上级目录
Expand-Archive -Path images.zip -DestinationPath ../datasets

# 移动JSON文件
Move-Item -Path *.json -Destination ../datasets

# 清理临时目录
cd ..
Remove-Item -Recurse -Force temp_evalmuse
