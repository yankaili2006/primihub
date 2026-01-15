#!/bin/bash

# Log cleanup script for PrimiHub
# This script helps manage log file sizes by truncating or rotating them

LOG_DIR="."
MAX_SIZE_MB=10
MAX_LINES=5000
KEEP_FIRST=1000
KEEP_LAST=4000

# Function to truncate a log file
truncate_log() {
    local file="$1"
    local max_lines="$2"
    local keep_first="$3"
    local keep_last="$4"
    
    if [[ ! -f "$file" ]]; then
        echo "File $file does not exist"
        return 1
    fi
    
    local total_lines=$(wc -l < "$file")
    
    if [[ $total_lines -le $max_lines ]]; then
        echo "File $file has $total_lines lines (<= $max_lines), no truncation needed"
        return 0
    fi
    
    echo "Truncating $file: $total_lines -> $((keep_first + keep_last)) lines"
    
    # Create temporary file
    local temp_file="${file}.tmp"
    
    # Keep first lines
    head -n "$keep_first" "$file" > "$temp_file"
    
    # Add truncation notice
    echo "" >> "$temp_file"
    echo "[LOG TRUNCATED - $((total_lines - keep_first - keep_last)) LINES REMOVED]" >> "$temp_file"
    echo "" >> "$temp_file"
    
    # Keep last lines
    tail -n "$keep_last" "$file" >> "$temp_file"
    
    # Replace original file
    mv "$temp_file" "$file"
    
    echo "Successfully truncated $file"
    return 0
}

# Function to check and rotate logs by size
rotate_log_by_size() {
    local file="$1"
    local max_size_mb="$2"
    
    if [[ ! -f "$file" ]]; then
        return 0
    fi
    
    local file_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
    local max_size_bytes=$((max_size_mb * 1024 * 1024))
    
    if [[ $file_size -gt $max_size_bytes ]]; then
        echo "Rotating $file (size: $((file_size/1024/1024))MB > ${max_size_mb}MB)"
        
        # Rotate existing backups
        for i in {9..1}; do
            if [[ -f "${file}.${i}" ]]; then
                mv "${file}.${i}" "${file}.$((i+1))" 2>/dev/null
            fi
        done
        
        # Move current log to .1
        mv "$file" "${file}.1"
        
        # Create new empty log file
        touch "$file"
        echo "Log rotation completed for $file"
    fi
}

# Main execution
main() {
    local action="${1:-truncate}"  # truncate or rotate
    
    case "$action" in
        "truncate")
            echo "Truncating log files in $LOG_DIR..."
            for log_file in "$LOG_DIR"/*.log "$LOG_DIR"/*log_node*; do
                if [[ -f "$log_file" ]]; then
                    truncate_log "$log_file" "$MAX_LINES" "$KEEP_FIRST" "$KEEP_LAST"
                fi
            done
            ;;
        "rotate")
            echo "Rotating log files in $LOG_DIR by size..."
            for log_file in "$LOG_DIR"/*.log "$LOG_DIR"/*log_node*; do
                if [[ -f "$log_file" ]]; then
                    rotate_log_by_size "$log_file" "$MAX_SIZE_MB"
                fi
            done
            ;;
        "clean")
            echo "Cleaning up old log files..."
            find "$LOG_DIR" -name "*.log.*" -mtime +7 -delete
            echo "Old log files cleaned up"
            ;;
        *)
            echo "Usage: $0 {truncate|rotate|clean}"
            echo "  truncate - Truncate log files by line count"
            echo "  rotate   - Rotate log files by size"
            echo "  clean    - Clean up old rotated log files"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"