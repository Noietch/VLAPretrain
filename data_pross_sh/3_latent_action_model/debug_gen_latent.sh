#!/bin/bash

echo "================================"
echo "Debug Script for gen_latent_lapa_video_ddp.py"
echo "================================"
echo ""

# Check if running in distributed mode
if [ -n "$RANK" ] && [ -n "$WORLD_SIZE" ]; then
    echo "Running in DISTRIBUTED mode"
    echo "  RANK: $RANK"
    echo "  WORLD_SIZE: $WORLD_SIZE"
    echo "  LOCAL_RANK: $LOCAL_RANK"
else
    echo "Running in NON-DISTRIBUTED mode"
fi

echo ""
echo "Running gen_latent_lapa_video_ddp.py with debug output..."
echo "================================"
echo ""

# Run the script with full output
python latent_action_model/gen_latent_lapa_video_ddp.py "$@" 2>&1 | tee gen_latent_debug.log

echo ""
echo "================================"
echo "Execution completed. Log saved to gen_latent_debug.log"
echo ""
echo "Checking for saved .pt files..."
find extract_data/latent_action -name "*.pt" -type f -mmin -5 2>/dev/null | head -20
echo ""
echo "Total .pt files in output directory:"
find extract_data/latent_action -name "*.pt" -type f 2>/dev/null | wc -l
