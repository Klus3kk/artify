#!/usr/bin/env python3
"""
Artify Server Runner
Production-ready script to start Artify services
"""

import os
import sys
import subprocess
import signal
import time
import argparse
from pathlib import Path
from typing import Optional, List
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("artify-runner")

class ArtifyServerRunner:
    """Manages Artify server processes"""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or os.getcwd())
        self.processes: List[subprocess.Popen] = []
        self.setup_environment()
    
    def setup_environment(self):
        """Setup environment variables and paths"""
        # Add project root to Python path
        sys.path.insert(0, str(self.project_root))
        
        # Set environment variables
        os.environ.setdefault('PYTHONPATH', str(self.project_root))
        os.environ.setdefault('TORCH_DISABLE_CLASSES', '1')
        os.environ.setdefault('STREAMLIT_TORCH_COMPAT', '1')
        
        # Create required directories
        required_dirs = [
            'logs',
            'api_output',
            'temp',
            'models',
            'images/output'
        ]
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Environment setup complete. Project root: {self.project_root}")
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        required_packages = [
            'fastapi',
            'uvicorn',
            'streamlit',
            'torch',
            'torchvision',
            'PIL'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            logger.error("Please install them using: pip install -r requirements.txt")
            return False
        
        logger.info("All required dependencies are installed")
        return True
    
    def start_fastapi(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = True) -> subprocess.Popen:
        """Start FastAPI server"""
        logger.info(f"Starting FastAPI server on {host}:{port}")
        
        cmd = [
            sys.executable, "-m", "uvicorn",
            "api.FastAPIHandler:app",
            "--host", host,
            "--port", str(port),
            "--log-level", "info"
        ]
        
        if reload:
            cmd.append("--reload")
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes.append(process)
            logger.info(f"FastAPI server started with PID: {process.pid}")
            return process
            
        except Exception as e:
            logger.error(f"Failed to start FastAPI server: {e}")
            raise
    
    def start_streamlit(self, port: int = 8501) -> subprocess.Popen:
        """Start Streamlit UI"""
        logger.info(f"Starting Streamlit UI on port {port}")
        
        ui_script = self.project_root / "interface" / "UIHandler.py"
        
        if not ui_script.exists():
            raise FileNotFoundError(f"UI script not found: {ui_script}")
        
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(ui_script),
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.processes.append(process)
            logger.info(f"Streamlit UI started with PID: {process.pid}")
            return process
            
        except Exception as e:
            logger.error(f"Failed to start Streamlit UI: {e}")
            raise
    
    def start_api_only(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
        """Start only the FastAPI server"""
        if not self.check_dependencies():
            return False
        
        try:
            fastapi_process = self.start_fastapi(host, port, reload)
            
            logger.info("=" * 60)
            logger.info("🚀 Artify API Server Started Successfully!")
            logger.info(f"📡 API Endpoint: http://{host}:{port}")
            logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
            logger.info("=" * 60)
            
            # Monitor the process
            self.monitor_processes([fastapi_process])
            
        except KeyboardInterrupt:
            logger.info("Shutting down API server...")
            self.shutdown()
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            self.shutdown()
            return False
        
        return True
    
    def start_ui_only(self, port: int = 8501):
        """Start only the Streamlit UI"""
        if not self.check_dependencies():
            return False
        
        try:
            streamlit_process = self.start_streamlit(port)
            
            logger.info("=" * 60)
            logger.info("🎨 Artify UI Started Successfully!")
            logger.info(f"🌐 Web Interface: http://localhost:{port}")
            logger.info("=" * 60)
            
            # Monitor the process
            self.monitor_processes([streamlit_process])
            
        except KeyboardInterrupt:
            logger.info("Shutting down UI...")
            self.shutdown()
        except Exception as e:
            logger.error(f"Failed to start UI: {e}")
            self.shutdown()
            return False
        
        return True
    
    def start_full_stack(self, api_port: int = 8000, ui_port: int = 8501, reload: bool = True):
        """Start both FastAPI and Streamlit servers"""
        if not self.check_dependencies():
            return False
        
        try:
            # Start FastAPI first
            fastapi_process = self.start_fastapi("0.0.0.0", api_port, reload)
            time.sleep(2)  # Give FastAPI time to start
            
            # Start Streamlit
            streamlit_process = self.start_streamlit(ui_port)
            
            logger.info("=" * 60)
            logger.info("🚀 Artify Full Stack Started Successfully!")
            logger.info(f"📡 API Endpoint: http://localhost:{api_port}")
            logger.info(f"📚 API Documentation: http://localhost:{api_port}/docs")
            logger.info(f"🌐 Web Interface: http://localhost:{ui_port}")
            logger.info("=" * 60)
            logger.info("Press Ctrl+C to stop all services")
            
            # Monitor both processes
            self.monitor_processes([fastapi_process, streamlit_process])
            
        except KeyboardInterrupt:
            logger.info("Shutting down all services...")
            self.shutdown()
        except Exception as e:
            logger.error(f"Failed to start full stack: {e}")
            self.shutdown()
            return False
        
        return True
    
    def monitor_processes(self, processes: List[subprocess.Popen]):
        """Monitor running processes and handle output"""
        try:
            while True:
                for process in processes:
                    if process.poll() is not None:
                        logger.error(f"Process {process.pid} has terminated unexpectedly")
                        self.shutdown()
                        return
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            raise
    
    def shutdown(self):
        """Shutdown all running processes"""
        logger.info("Shutting down Artify services...")
        
        for process in self.processes:
            try:
                if process.poll() is None:  # Process still running
                    logger.info(f"Terminating process {process.pid}")
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Force killing process {process.pid}")
                        process.kill()
                        process.wait()
                        
            except Exception as e:
                logger.error(f"Error shutting down process {process.pid}: {e}")
        
        self.processes.clear()
        logger.info("All services stopped")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Artify Server Runner")
    parser.add_argument("--mode", choices=["api", "ui", "full"], default="full",
                      help="Service mode to run")
    parser.add_argument("--api-port", type=int, default=8000,
                      help="FastAPI server port")
    parser.add_argument("--ui-port", type=int, default=8501,
                      help="Streamlit UI port")
    parser.add_argument("--no-reload", action="store_true",
                      help="Disable auto-reload for FastAPI")
    parser.add_argument("--project-root", type=str,
                      help="Project root directory")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ArtifyServerRunner(args.project_root)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Received interrupt signal")
        runner.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start services based on mode
    success = False
    reload = not args.no_reload
    
    if args.mode == "api":
        success = runner.start_api_only("0.0.0.0", args.api_port, reload)
    elif args.mode == "ui":
        success = runner.start_ui_only(args.ui_port)
    elif args.mode == "full":
        success = runner.start_full_stack(args.api_port, args.ui_port, reload)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()