#!/usr/bin/env python3
"""
Continuous GPM metrics monitoring wrapper

Runs gpm_metrics_reader repeatedly and outputs CSV format for plotting.

Usage:
    python gpm_monitor.py [options]

Options:
    --interval MSEC     Sampling interval in milliseconds (default: 1000)
    --duration SEC      Total duration in seconds (default: infinite)
    --output FILE       Output CSV file (default: stdout)
    --binary PATH       Path to gpm_metrics_reader binary (default: ./gpm_metrics_reader)
"""

import subprocess
import sys
import time
import re
import argparse
import signal
from datetime import datetime


class GPMMonitor:
    def __init__(self, binary_path, interval_ms, output_file=None):
        self.binary_path = binary_path
        self.interval_ms = interval_ms
        self.interval_sec = interval_ms / 1000.0
        self.output_file = output_file
        self.running = True
        self.header_written = False
        
        # Setup signal handler for clean exit
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        sys.stderr.write("\n\nStopping monitoring...\n")
        self.running = False
    
    def _parse_output(self, output):
        """Parse gpm_metrics_reader output and extract metrics"""
        lines = output.strip().split('\n')
        
        # Find device info
        device_id = None
        device_name = None
        gpu_instance_id = None
        compute_instance_id = None
        
        for line in lines:
            if line.startswith('GPU '):
                match = re.match(r'GPU (\d+): (.+)', line)
                if match:
                    device_id = match.group(1)
                    device_name = match.group(2)
            elif 'MIG Slice' in line:
                match = re.search(r'GI: (\d+), CI: (\d+)', line)
                if match:
                    gpu_instance_id = match.group(1)
                    compute_instance_id = match.group(2)
        
        # Parse metrics table
        metrics = []
        in_table = False
        
        for line in lines:
            # Skip header and separator lines
            if 'ID' in line and 'Name' in line and 'Value' in line:
                in_table = True
                continue
            if '-----' in line:
                continue
            
            if in_table and line.strip():
                # Parse metric line: ID, Name, Value, Unit, Status
                parts = line.split()
                if len(parts) >= 5 and parts[0].isdigit():
                    try:
                        metric_id = parts[0]
                        # Name could be multiple words, find where value starts
                        value_idx = -1
                        for i, part in enumerate(parts[1:], 1):
                            try:
                                float(part)
                                value_idx = i
                                break
                            except ValueError:
                                continue
                        
                        if value_idx > 0:
                            metric_name = '_'.join(parts[1:value_idx])
                            metric_value = parts[value_idx]
                            metric_unit = parts[value_idx + 1] if value_idx + 1 < len(parts) else ''
                            metric_status = parts[-1]
                            
                            if metric_status == 'OK':
                                metrics.append({
                                    'id': metric_id,
                                    'name': metric_name,
                                    'value': metric_value,
                                    'unit': metric_unit
                                })
                    except (ValueError, IndexError):
                        continue
        
        return {
            'device_id': device_id or '0',
            'device_name': device_name or 'Unknown',
            'gpu_instance_id': gpu_instance_id or '',
            'compute_instance_id': compute_instance_id or '',
            'metrics': metrics
        }
    
    def _write_csv_header(self, out):
        """Write CSV header"""
        out.write("timestamp,device_id,device_name,gpu_instance_id,compute_instance_id,")
        out.write("metric_id,metric_name,value,unit\n")
        out.flush()
    
    def _write_csv_row(self, out, timestamp, data):
        """Write CSV rows for all metrics"""
        for metric in data['metrics']:
            out.write(f"{timestamp},{data['device_id']},{data['device_name']},")
            out.write(f"{data['gpu_instance_id']},{data['compute_instance_id']},")
            out.write(f"{metric['id']},{metric['name']},{metric['value']},{metric['unit']}\n")
        out.flush()
    
    def run(self, duration_sec=None):
        """Run continuous monitoring"""
        start_time = time.time()
        
        # Open output file or use stdout
        if self.output_file:
            out = open(self.output_file, 'w')
            sys.stderr.write(f"Writing to {self.output_file}\n")
        else:
            out = sys.stdout
        
        try:
            # Write CSV header
            self._write_csv_header(out)
            
            iteration = 0
            while self.running:
                # Check duration
                if duration_sec and (time.time() - start_time) >= duration_sec:
                    sys.stderr.write(f"\nReached duration limit of {duration_sec}s\n")
                    break
                
                iter_start = time.time()
                
                # Run gpm_metrics_reader
                try:
                    result = subprocess.run(
                        [self.binary_path],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        # Parse output
                        data = self._parse_output(result.stdout)
                        
                        # Get timestamp
                        timestamp = datetime.now().isoformat()
                        
                        # Write CSV rows
                        self._write_csv_row(out, timestamp, data)
                        
                        iteration += 1
                        if iteration % 10 == 0:
                            sys.stderr.write(f"\rSamples collected: {iteration}")
                            sys.stderr.flush()
                    else:
                        sys.stderr.write(f"\nError running {self.binary_path}: {result.stderr}\n")
                        break
                
                except subprocess.TimeoutExpired:
                    sys.stderr.write(f"\nTimeout running {self.binary_path}\n")
                    continue
                except FileNotFoundError:
                    sys.stderr.write(f"\nBinary not found: {self.binary_path}\n")
                    break
                
                # Sleep for remaining interval time
                elapsed = time.time() - iter_start
                sleep_time = max(0, self.interval_sec - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        finally:
            if self.output_file and out != sys.stdout:
                out.close()
            sys.stderr.write(f"\n\nTotal samples collected: {iteration}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Continuous GPM metrics monitoring wrapper',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=1000,
        help='Sampling interval in milliseconds (default: 1000)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Total duration in seconds (default: infinite)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file (required)'
    )
    
    parser.add_argument(
        '--binary',
        type=str,
        default='./gpm_metrics_reader',
        help='Path to gpm_metrics_reader binary (default: ./gpm_metrics_reader)'
    )
    
    args = parser.parse_args()
    
    # Validate interval
    if args.interval < 150:
        sys.stderr.write("Warning: interval < 150ms may be too fast (GPM sampling needs >100ms)\n")
    
    # Create and run monitor
    monitor = GPMMonitor(args.binary, args.interval, args.output)
    
    sys.stderr.write(f"Starting GPM monitoring...\n")
    sys.stderr.write(f"  Binary: {args.binary}\n")
    sys.stderr.write(f"  Interval: {args.interval}ms\n")
    sys.stderr.write(f"  Duration: {args.duration}s\n" if args.duration else "  Duration: infinite\n")
    sys.stderr.write(f"  Output: {args.output}\n" if args.output else "  Output: stdout\n")
    sys.stderr.write(f"\nPress Ctrl+C to stop\n\n")
    
    monitor.run(args.duration)


if __name__ == '__main__':
    main()