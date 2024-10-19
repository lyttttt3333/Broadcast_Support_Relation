#!/bin/bash


PARENT_DIR="/path/to/data_interaction/train"    
PYTHON_COMMAND="/paty/to/python.sh"   
PYTHON_SCRIPT="/path/to/data_collection/gen_grasp.py"

for dir in "$PARENT_DIR"/*/; do

    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        
        last4="${dirname: -4}"

        if [ ${#dirname} -ge 4 ]; then
            echo "运行 Python 脚本，参数: $last4"
            "$PYTHON_COMMAND" "$PYTHON_SCRIPT" --indice "$last4"
        else
            echo "跳过 $dirname: 名称长度不足4个字符。"
        fi
    fi
done
