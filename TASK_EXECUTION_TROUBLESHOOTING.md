# Task Execution Troubleshooting Guide

This guide helps diagnose and fix common task execution issues in PrimiHub.

## Common Issues and Solutions

### 1. Port Conflicts
**Symptoms:**
- "Address already in use" errors
- Nodes fail to start
- Task execution hangs

**Solutions:**
```bash
# Check for port conflicts
netstat -tulpn | grep -E "50050|50051|50052|9099"

# Kill conflicting processes
pkill -f "primihub-node"
pkill -f "task_main"

# Alternative: Use different ports
# Edit config files to use different port numbers
```

### 2. Task Execution Failures
**Symptoms:**
- Tasks return "FAIL" status
- Python execution errors
- Timeout errors

**Solutions:**
```bash
# Check Python environment
python3 -c "import primihub"

# Install Python dependencies
cd python && pip install -e . && cd ..

# Check task logs
tail -f log_node0
grep -i "error\|fail\|exception" log_node*

# Use safe execution wrapper
./scripts/execute_task_safe.sh test
```

### 3. Resource Management Issues
**Symptoms:**
- Tasks don't clean up properly
- Memory leaks
- Orphaned processes

**Solutions:**
```bash
# Clean up orphaned processes
./scripts/cleanup_logs.sh clean

# Check running processes
ps aux | grep -i primihub

# Monitor resource usage
top -p $(pgrep -f "primihub-node" | tr '\n' ',' | sed 's/,$//')
```

### 4. Configuration Issues
**Symptoms:**
- Nodes can't communicate
- Dataset not found errors
- Invalid configuration

**Solutions:**
```bash
# Verify configuration files
./scripts/fix_task_execution.sh

# Check data files exist
ls -la data/*.csv

# Validate network connectivity
ping -c 3 127.0.0.1
```

## Diagnostic Commands

### System Health Check
```bash
# Run comprehensive health check
./scripts/check_task_health.sh

# Check disk space
df -h .

# Check memory usage
free -h
```

### Log Analysis
```bash
# Check for recent errors
grep -c "ERROR\|FAIL" log_node* | tail -5

# Monitor real-time logs
tail -f log_node0 | grep -E "ERROR|WARNING|INFO"

# Search for specific task errors
grep "task_id_200" log_node* | grep -i error
```

### Network Diagnostics
```bash
# Check if ports are listening
for port in 50050 50051 50052 9099; do
    echo "Port $port: $(netstat -tulpn | grep -c ":$port ")"
done

# Test connectivity
telnet 127.0.0.1 50050
```

## Quick Fix Scripts

### Reset Environment
```bash
# Stop all processes and clean up
./scripts/fix_task_execution.sh

# Restart cluster
./start_local_cluster.sh
```

### Safe Task Execution
```bash
# Execute task with timeout and logging
./scripts/execute_task_safe.sh <task_type>

# Monitor execution
tail -f log/task_*.log
```

### Log Management
```bash
# Truncate large log files
./scripts/cleanup_logs.sh truncate

# Rotate logs by size
./scripts/cleanup_logs.sh rotate

# Clean old logs
./scripts/cleanup_logs.sh clean
```

## Common Error Messages and Fixes

### "Address already in use"
- **Cause:** Another process using the port
- **Fix:** Kill existing processes or change ports

### "Task execution failed"
- **Cause:** Python execution error or timeout
- **Fix:** Check Python environment and increase timeout

### "Dataset not found"
- **Cause:** Missing data files
- **Fix:** Verify data files exist in correct location

### "Connection refused"
- **Cause:** Node not running or network issue
- **Fix:** Restart nodes and check network

## Prevention Best Practices

1. **Regular Maintenance**
   - Run health checks periodically
   - Clean up old logs and temporary files
   - Update dependencies regularly

2. **Monitoring**
   - Set up log monitoring
   - Monitor system resources
   - Use the health check script

3. **Testing**
   - Test task execution regularly
   - Validate configuration changes
   - Use the safe execution wrapper

## Emergency Recovery

If tasks are consistently failing:

1. **Stop all processes:**
   ```bash
   pkill -f primihub
   pkill -f task_main
   ```

2. **Clean environment:**
   ```bash
   ./scripts/fix_task_execution.sh
   ```

3. **Restart fresh:**
   ```bash
   ./start_local_cluster.sh
   ```

4. **Test basic functionality:**
   ```bash
   ./scripts/execute_task_safe.sh test
   ```