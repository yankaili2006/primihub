#!/bin/bash
# Health check for PrimiHub task execution

echo "=== PrimiHub Task Health Check ==="

# Check if nodes are running
NODES_RUNNING=$(ps aux | grep -c "primihub-node")
echo "PrimiHub nodes running: $((NODES_RUNNING - 1))"

# Check ports
PORTS="50050 50051 50052"
for port in $PORTS; do
    if netstat -tulpn | grep -q ":$port "; then
        echo "✓ Port $port is listening"
    else
        echo "✗ Port $port is not listening"
    fi
done

# Check recent task errors
RECENT_ERRORS=$(grep -c "ERROR\|FAIL" log_node* 2>/dev/null | tail -1 || echo "0")
echo "Recent errors in logs: $RECENT_ERRORS"

# Check disk space
DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
echo "Available disk space: $DISK_SPACE"

# Check if task_main exists
if [ -f "task_main" ]; then
    echo "✓ task_main executable exists"
else
    echo "✗ task_main executable missing"
fi

# Check Python environment
if python3 -c "import primihub" 2>/dev/null; then
    echo "✓ Python primihub module available"
else
    echo "✗ Python primihub module not available"
fi

echo "=== Health Check Complete ==="