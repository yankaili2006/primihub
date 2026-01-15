# Log Management for PrimiHub

This document describes the log management solutions implemented to prevent log files from growing too large and exceeding model limits.

## Overview

PrimiHub now includes multiple layers of log management:

1. **C++ Log Truncation** - In-memory log truncation in the custom Log class
2. **Python Log Rotation** - File-based log rotation in Python logging handlers
3. **Utility Scripts** - Manual log cleanup and truncation tools

## Implemented Solutions

### 1. C++ Log Truncation

**File**: `src/primihub/util/log.h`

- Added automatic truncation to the `Log` class
- Maximum message limit: 10,000 messages
- Preserves first 1,000 and last 9,000 messages when truncating
- Adds truncation notice when messages are removed

### 2. Python Log Rotation

**File**: `python/primihub/utils/logger_util.py`

- Enhanced `FLFileHandler` with log rotation
- Maximum file size: 5MB
- Maximum backup files: 5
- Automatic rotation when file size exceeds limit

### 3. Utility Scripts

#### cleanup_logs.sh
**Location**: `scripts/cleanup_logs.sh`

A bash script for manual log management:

```bash
# Truncate logs by line count (default: 5000 lines)
./scripts/cleanup_logs.sh truncate

# Rotate logs by file size (default: 10MB)
./scripts/cleanup_logs.sh rotate

# Clean up old rotated logs (older than 7 days)
./scripts/cleanup_logs.sh clean
```

#### truncate_logs.py
**Location**: `scripts/truncate_logs.py`

A Python script with more advanced truncation options:

```bash
python3 scripts/truncate_logs.py --log-dir ./log --max-lines 5000
```

## Configuration

### C++ Logging (glog)

Current glog configuration in `src/primihub/node/main.cc`:

```cpp
FLAGS_max_log_size = 10; // 10MB per log file
FLAGS_stop_logging_if_full_disk = true;
```

### Python Logging

Default configuration in `python/primihub/utils/logger_util.py`:

```python
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5MB
MAX_LOG_FILES = 5               # Keep 5 backup files
```

## Usage Examples

### Manual Log Truncation

```bash
# Truncate all log files in current directory
cd /home/primihub/github/primihub
./scripts/cleanup_logs.sh truncate

# Truncate with custom parameters
./scripts/cleanup_logs.sh truncate log_dir=./log max_lines=3000
```

### Automated Log Rotation

The Python logging system now automatically rotates logs when they exceed 5MB in size, keeping up to 5 backup files.

### Monitoring Log Sizes

```bash
# Check current log sizes
ls -la log_node*
wc -l log_node*

# Check disk usage
du -sh log/
```

## Best Practices

1. **Regular Monitoring**: Check log sizes periodically
2. **Automated Cleanup**: Set up cron jobs for regular log rotation
3. **Size Limits**: Adjust `MAX_LOG_SIZE` based on available disk space
4. **Backup Strategy**: Keep sufficient backup files for debugging

## Troubleshooting

If logs continue to grow too large:

1. Check if the log rotation is working
2. Verify file permissions for log rotation
3. Monitor disk space regularly
4. Consider reducing log verbosity for non-critical operations

## Future Improvements

- Implement log compression for archived logs
- Add log level filtering to reduce verbosity
- Create centralized log management service
- Add log analytics and alerting