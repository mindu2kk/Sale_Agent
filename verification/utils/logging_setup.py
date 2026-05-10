"""
Logging Setup Utilities cho Verification Agent

Centralized logging configuration và initialization:
- Environment-specific logging setup
- Configuration file loading
- Logger factory methods
- Performance monitoring integration
"""

import os
import logging
import logging.config
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from ..config import VerificationConfig, get_config
from .logging import (
    setup_verification_logger, 
    EnhancedVerificationLogger,
    configure_root_logger,
    get_observability_manager,
    CorrelationIDGenerator
)


class LoggingConfigurator:
    """
    Centralized logging configuration manager
    """
    
    def __init__(self, config: Optional[VerificationConfig] = None):
        """
        Initialize logging configurator
        
        Args:
            config: Verification configuration (uses global if None)
        """
        self.config = config or get_config()
        self._configured = False
        self._loggers: Dict[str, EnhancedVerificationLogger] = {}
    
    def setup_logging(self, environment: Optional[str] = None) -> None:
        """
        Setup comprehensive logging system
        
        Args:
            environment: Environment name (development, production, testing)
        """
        
        if self._configured:
            return
        
        # Determine environment
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "development").lower()
        
        # Load logging configuration
        logging_config = self._load_logging_config(environment)
        
        # Create logs directory
        self._ensure_logs_directory()
        
        # Configure logging
        try:
            logging.config.dictConfig(logging_config["logging"])
        except Exception as e:
            print(f"Warning: Failed to configure logging from config: {e}")
            self._setup_fallback_logging()
        
        # Configure root logger
        configure_root_logger(self.config)
        
        # Setup observability manager
        observability_manager = get_observability_manager(self.config)
        
        self._configured = True
        
        # Log successful setup
        logger = self.get_logger("verification.setup")
        log_level_value = self.config.log_level.value if hasattr(self.config.log_level, 'value') else self.config.log_level
        logger.info(
            f"Logging system configured for environment: {environment}",
            environment=environment,
            log_level=log_level_value,
            detailed_logging=self.config.detailed_logging
        )
    
    def _load_logging_config(self, environment: str) -> Dict[str, Any]:
        """Load logging configuration for environment"""
        
        # Try environment-specific config first
        config_paths = [
            f"verification/config/environments/{environment}.yaml",
            "verification/config/logging_config.yaml",
            "verification/config/verification_config.yaml"
        ]
        
        for config_path in config_paths:
            config_file = Path(config_path)
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                    
                    # Extract logging configuration
                    if "logging" in config_data:
                        return config_data
                    
                except Exception as e:
                    print(f"Warning: Failed to load config from {config_path}: {e}")
                    continue
        
        # Return default configuration
        return self._get_default_logging_config(environment)
    
    def _get_default_logging_config(self, environment: str) -> Dict[str, Any]:
        """Get default logging configuration for environment"""
        
        log_level = "DEBUG" if environment == "development" else "INFO"
        if environment == "testing":
            log_level = "ERROR"
        
        return {
            "logging": {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "structured_json": {
                        "class": "verification.utils.logging.StructuredFormatter"
                    },
                    "detailed_console": {
                        "format": "[%(asctime)s] %(levelname)-8s | %(name)-20s | %(message)s",
                        "datefmt": "%Y-%m-%d %H:%M:%S"
                    }
                },
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "level": log_level,
                        "formatter": "detailed_console",
                        "stream": "ext://sys.stdout"
                    },
                    "file": {
                        "class": "logging.handlers.RotatingFileHandler",
                        "level": "DEBUG",
                        "formatter": "structured_json",
                        "filename": "logs/verification_workflow.log",
                        "maxBytes": 10485760,  # 10MB
                        "backupCount": 5,
                        "encoding": "utf8"
                    }
                },
                "loggers": {
                    "verification": {
                        "level": log_level,
                        "handlers": ["console", "file"],
                        "propagate": False
                    },
                    "verification.workflow": {
                        "level": log_level,
                        "handlers": ["console", "file"],
                        "propagate": False
                    },
                    "verification.metrics": {
                        "level": "INFO",
                        "handlers": ["file"],
                        "propagate": False
                    }
                }
            }
        }
    
    def _ensure_logs_directory(self) -> None:
        """Ensure logs directory exists"""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different log types
        (logs_dir / "workflow").mkdir(exist_ok=True)
        (logs_dir / "metrics").mkdir(exist_ok=True)
        (logs_dir / "errors").mkdir(exist_ok=True)
    
    def _setup_fallback_logging(self) -> None:
        """Setup fallback logging configuration"""
        log_level_value = self.config.log_level.value if hasattr(self.config.log_level, 'value') else self.config.log_level
        logging.basicConfig(
            level=getattr(logging, log_level_value),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("logs/verification_fallback.log")
            ]
        )
    
    def get_logger(self, name: str) -> EnhancedVerificationLogger:
        """
        Get or create enhanced verification logger
        
        Args:
            name: Logger name
            
        Returns:
            EnhancedVerificationLogger instance
        """
        
        if not self._configured:
            self.setup_logging()
        
        if name not in self._loggers:
            self._loggers[name] = setup_verification_logger(name, self.config)
        
        return self._loggers[name]
    
    def create_workflow_logger(self, workflow_id: str) -> EnhancedVerificationLogger:
        """
        Create logger specifically for a workflow
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            EnhancedVerificationLogger configured for workflow
        """
        
        logger_name = f"verification.workflow.{workflow_id}"
        logger = self.get_logger(logger_name)
        
        # Generate correlation ID for workflow
        correlation_id = CorrelationIDGenerator.generate_correlation_id()
        
        return logger
    
    def get_performance_logger(self) -> EnhancedVerificationLogger:
        """Get logger for performance metrics"""
        return self.get_logger("verification.metrics")
    
    def get_error_logger(self) -> EnhancedVerificationLogger:
        """Get logger for error handling"""
        return self.get_logger("verification.errors")
    
    def get_workflow_logger(self) -> EnhancedVerificationLogger:
        """Get logger for workflow execution"""
        return self.get_logger("verification.workflow")


# Global configurator instance
_logging_configurator: Optional[LoggingConfigurator] = None


def get_logging_configurator(config: Optional[VerificationConfig] = None) -> LoggingConfigurator:
    """Get global logging configurator instance"""
    global _logging_configurator
    
    if _logging_configurator is None:
        _logging_configurator = LoggingConfigurator(config)
    
    return _logging_configurator


def setup_verification_logging(environment: Optional[str] = None, 
                             config: Optional[VerificationConfig] = None) -> LoggingConfigurator:
    """
    Setup verification logging system
    
    Args:
        environment: Environment name (development, production, testing)
        config: Verification configuration
        
    Returns:
        Configured LoggingConfigurator instance
    """
    
    configurator = get_logging_configurator(config)
    configurator.setup_logging(environment)
    return configurator


def get_workflow_logger(workflow_id: Optional[str] = None) -> EnhancedVerificationLogger:
    """
    Get workflow logger với optional workflow ID
    
    Args:
        workflow_id: Optional workflow identifier
        
    Returns:
        EnhancedVerificationLogger for workflow
    """
    
    configurator = get_logging_configurator()
    
    if workflow_id:
        return configurator.create_workflow_logger(workflow_id)
    else:
        return configurator.get_workflow_logger()


def get_verification_logger(component: str = "verification") -> EnhancedVerificationLogger:
    """
    Get verification logger for specific component
    
    Args:
        component: Component name (verification, agent, workflow, etc.)
        
    Returns:
        EnhancedVerificationLogger for component
    """
    
    configurator = get_logging_configurator()
    return configurator.get_logger(f"verification.{component}")


class LoggingContextManager:
    """
    Context manager cho logging setup và cleanup
    """
    
    def __init__(self, environment: Optional[str] = None, 
                 config: Optional[VerificationConfig] = None):
        """
        Initialize logging context manager
        
        Args:
            environment: Environment name
            config: Verification configuration
        """
        self.environment = environment
        self.config = config
        self.configurator: Optional[LoggingConfigurator] = None
    
    def __enter__(self) -> LoggingConfigurator:
        """Setup logging and return configurator"""
        self.configurator = setup_verification_logging(self.environment, self.config)
        return self.configurator
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup logging resources"""
        if self.configurator:
            # Flush all handlers
            for logger_name, logger in self.configurator._loggers.items():
                for handler in logger.logger.handlers:
                    handler.flush()
            
            # Export any buffered metrics
            for logger_name, logger in self.configurator._loggers.items():
                if hasattr(logger, 'export_metrics'):
                    metrics = logger.export_metrics()
                    if metrics:
                        print(f"Exported {len(metrics)} metrics from {logger_name}")


def configure_logging_for_testing() -> LoggingConfigurator:
    """
    Configure minimal logging for testing
    
    Returns:
        LoggingConfigurator configured for testing
    """
    
    # Create minimal config for testing
    from ..config import VerificationConfig, LogLevel
    test_config = VerificationConfig(
        log_level=LogLevel.ERROR,
        detailed_logging=False,
        performance_tracking=False
    )
    
    configurator = LoggingConfigurator(test_config)
    configurator.setup_logging("testing")
    
    return configurator


def export_logging_metrics(output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Export logging metrics và statistics
    
    Args:
        output_path: Optional path to save metrics
        
    Returns:
        Dictionary containing logging metrics
    """
    
    configurator = get_logging_configurator()
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "loggers_configured": len(configurator._loggers),
        "logger_names": list(configurator._loggers.keys()),
        "configuration_status": configurator._configured,
        "log_level": configurator.config.log_level.value,
        "detailed_logging": configurator.config.detailed_logging
    }
    
    # Collect metrics from individual loggers
    logger_metrics = {}
    for logger_name, logger in configurator._loggers.items():
        if hasattr(logger, 'export_metrics'):
            logger_metrics[logger_name] = logger.export_metrics()
    
    metrics["logger_metrics"] = logger_metrics
    
    # Save to file if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    
    return metrics