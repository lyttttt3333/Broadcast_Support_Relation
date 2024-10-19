#!/bin/bash  
  

PARENT_DIR="/path/to/data_interaction/train"    
PYTHON_COMMAND="/paty/to/python.sh"   
PYTHON_SCRIPT="/path/to/data_collection/gen_interaction.py"

directory_path="/home/sim/safe_manipulation/rep/broadcast_final/data/data_scene"  
  
if [ ! -d "$directory_path" ]; then  
    echo "Error: Directory '$directory_path' does not exist."  
    exit 1  
fi  

count = 0
   
for file in "$directory_path"/*; do  
    if [ -f "$file" ]; then  
        ((count++))
        if [ "$count" -le 100 ]; then
            echo "interact '$file' now, idx '$count'." 
            PYTHON_COMMAND PYTHON_SCRIPT --structure_path "$file"  
        else
            break
        fi
    fi  
done  
  
echo "All files processed."
