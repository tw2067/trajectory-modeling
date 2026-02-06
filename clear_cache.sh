bash

#!/bin/bash

echo "Clearing all Python caches..."

# Remove all __pycache__ directories
find /home/gaga/tamarw1/trajectory-modeling -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Remove all .pyc files
find /home/gaga/tamarw1/trajectory-modeling -type f -name "*.pyc" -delete 2>/dev/null

# Remove all .pyo files
find /home/gaga/tamarw1/trajectory-modeling -type f -name "*.pyo" -delete 2>/dev/null

echo "Cache cleared!"