"""
Monitor agent execution and display step-by-step results.
This script watches the agent output and displays it in real-time.
"""

import sys
import time
import os
from pathlib import Path
from datetime import datetime

def monitor_agent_log(log_file="/tmp/agent_output.log"):
    """Monitor agent log file and display updates."""
    print("="*70)
    print("🔍 MONITORING AGENT EXECUTION")
    print("="*70)
    print(f"📁 Watching log file: {log_file}")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_size = 0
    try:
        while True:
            if os.path.exists(log_file):
                current_size = os.path.getsize(log_file)
                
                if current_size > last_size:
                    # Read new content
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        f.seek(last_size)
                        new_content = f.read()
                        
                        if new_content.strip():
                            print(new_content, end='', flush=True)
                    
                    last_size = current_size
                else:
                    time.sleep(1)
            else:
                print(f"⏳ Waiting for log file to be created...")
                time.sleep(2)
                
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error monitoring: {e}")


def check_outputs_dir(outputs_dir=None):
    """Check outputs directory for generated files."""
    if outputs_dir is None:
        outputs_dir = Path(__file__).parent / "outputs"
    
    outputs_dir = Path(outputs_dir)
    
    if not outputs_dir.exists():
        print(f"⚠️  Outputs directory not found: {outputs_dir}")
        return
    
    print("\n" + "="*70)
    print("📁 OUTPUTS DIRECTORY CONTENTS")
    print("="*70)
    print(f"Directory: {outputs_dir}\n")
    
    files = sorted(outputs_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not files:
        print("No output files found yet.")
        return
    
    for file in files[:10]:  # Show last 10 files
        if file.is_file():
            size = file.stat().st_size
            mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"📄 {file.name}")
            print(f"   Size: {size:,} bytes")
            print(f"   Modified: {mtime}")
            
            # Show preview for text files
            if file.suffix in ['.txt', '.log'] and size < 100000:  # < 100KB
                try:
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = content.split('\n')
                        print(f"   Preview (last 5 lines):")
                        for line in lines[-5:]:
                            if line.strip():
                                print(f"     {line[:100]}")
                except:
                    pass
            print()


def show_step_by_step(step_file=None):
    """Show step-by-step results from a specific file."""
    if step_file is None:
        outputs_dir = Path(__file__).parent / "outputs"
        step_files = sorted(outputs_dir.glob("agent_steps_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not step_files:
            print("❌ No step-by-step files found.")
            return
        
        step_file = step_files[0]
        print(f"📋 Showing steps from: {step_file.name}\n")
    
    step_file = Path(step_file)
    
    if not step_file.exists():
        print(f"❌ File not found: {step_file}")
        return
    
    with open(step_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        print("="*70)
        print("STEP-BY-STEP AGENT EXECUTION")
        print("="*70)
        print(content)


def main():
    """Main monitoring function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor agent execution")
    parser.add_argument("--log", default="/tmp/agent_output.log", help="Log file to monitor")
    parser.add_argument("--outputs", action="store_true", help="Check outputs directory")
    parser.add_argument("--steps", help="Show step-by-step from specific file")
    parser.add_argument("--watch", action="store_true", help="Watch log file in real-time")
    
    args = parser.parse_args()
    
    if args.steps:
        show_step_by_step(args.steps)
    elif args.outputs:
        check_outputs_dir()
    elif args.watch:
        monitor_agent_log(args.log)
    else:
        # Default: check outputs and watch log
        check_outputs_dir()
        print("\n")
        monitor_agent_log(args.log)


if __name__ == "__main__":
    main()

