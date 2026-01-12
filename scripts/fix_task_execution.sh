#!/bin/bash

# Task Execution Fix Script for PrimiHub
# This script addresses common task execution issues

set -e

echo "=== PrimiHub Task Execution Fix Script ==="

# 1. Stop any running PrimiHub processes
echo "1. Stopping existing PrimiHub processes..."
pkill -f "primihub-node" || true
pkill -f "task_main" || true
sleep 2

# 2. Check for port conflicts
echo "2. Checking for port conflicts..."
PORTS="50050 50051 50052 9099"
for port in $PORTS; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "WARNING: Port $port is in use"
        lsof -i :$port
    else
        echo "Port $port is available"
    fi
done

# 3. Clean up temporary files and logs
echo "3. Cleaning up temporary files..."
rm -f /home/primihub/.cache/bazel/_bazel_primihub/06b9b24d6441aa1cc8ff8d05342d337a/execroot/primihub/bazel-out/k8-fastbuild/bin/task_main_* || true
rm -f /tmp/task_* || true

# 4. Verify configuration files
echo "4. Verifying configuration files..."
CONFIG_FILES="config/primihub_node0_local.yaml config/primihub_node1_local.yaml config/primihub_node2_local.yaml"
for config in $CONFIG_FILES; do
    if [ -f "$config" ]; then
        echo "✓ $config exists"
        # Check for valid IP addresses
        if grep -q "127.0.0.1" "$config"; then
            echo "  ✓ Uses localhost IP (127.0.0.1)"
        else
            echo "  ⚠ Check IP configuration in $config"
        fi
    else
        echo "✗ $config missing"
    fi
done

# 5. Check Python environment
echo "5. Checking Python environment..."
if python3 -c "import primihub" 2>/dev/null; then
    echo "✓ Python primihub module available"
else
    echo "✗ Python primihub module not available"
    echo "  Installing Python dependencies..."
    cd python && pip install -e . && cd ..
fi

# 6. Verify data files exist
echo "6. Verifying data files..."
DATA_FILES="data/train_party_0.csv data/train_party_1.csv data/train_party_2.csv data/test_party_0.csv data/test_party_1.csv data/test_party_2.csv"
for data_file in $DATA_FILES; do
    if [ -f "$data_file" ]; then
        echo "✓ $data_file exists"
    else
        echo "✗ $data_file missing"
    fi
done

# 7. Build task_main if needed
echo "7. Building task_main if needed..."
if [ ! -f "task_main" ] || [ ! -f "simple_task_main.cc" ]; then
    echo "Building simple task_main..."
    g++ -o task_main simple_task_main.cc
    echo "✓ task_main built"
else
    echo "✓ task_main exists"
fi

# 8. Set up log directory
echo "8. Setting up log directory..."
mkdir -p log
chmod 755 log

# 9. Create task execution health check
echo "9. Creating health check script..."
cat > scripts/check_task_health.sh << 'EOF'
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

echo "=== Health Check Complete ==="
EOF
chmod +x scripts/check_task_health.sh

echo "✓ Health check script created"

# 10. Create task execution wrapper with better error handling
echo "10. Creating improved task execution wrapper..."
cat > scripts/execute_task_safe.sh << 'EOF'
#!/bin/bash
# Safe task execution wrapper with error handling

TASK_TYPE="${1:-test}"
TIMEOUT_SECONDS=300
LOG_DIR="log"
TASK_ID="task_$(date +%s)"

echo "[$(date)] Starting task execution: $TASK_TYPE (ID: $TASK_ID)"

# Function to cleanup on exit
cleanup() {
    echo "[$(date)] Cleaning up task $TASK_ID..."
    pkill -f "task_main" 2>/dev/null || true
    # Add any additional cleanup here
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Execute task with timeout
echo "[$(date)] Executing task with timeout ${TIMEOUT_SECONDS}s..."
if timeout $TIMEOUT_SECONDS ./task_main "$@" 2>&1 | tee "$LOG_DIR/${TASK_ID}.log"; then
    echo "[$(date)] Task $TASK_ID completed successfully"
    exit 0
else
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo "[$(date)] ERROR: Task $TASK_ID timed out after ${TIMEOUT_SECONDS}s"
    else
        echo "[$(date)] ERROR: Task $TASK_ID failed with exit code $EXIT_CODE"
    fi
    
    # Log the failure
    echo "TASK_FAILURE: $TASK_ID, TYPE: $TASK_TYPE, EXIT_CODE: $EXIT_CODE, TIMESTAMP: $(date)" >> "$LOG_DIR/task_failures.log"
    
    exit $EXIT_CODE
fi
EOF
chmod +x scripts/execute_task_safe.sh

echo "✓ Safe task execution wrapper created"

echo ""
echo "=== Fixes Applied Successfully ==="
echo "Next steps:"
echo "1. Run: ./scripts/check_task_health.sh"
echo "2. Start nodes: ./start_local_cluster.sh"
echo "3. Test task: ./scripts/execute_task_safe.sh test"
echo "4. Monitor logs: tail -f log_node0"