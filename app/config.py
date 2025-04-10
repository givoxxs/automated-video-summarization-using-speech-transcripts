from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

class Config:
    # Default paths configuration
    BASE_DIR = Path("./data")
    PATHS = {
        "video": BASE_DIR / "video",
        "audio": BASE_DIR / "audio", 
        "transcript": BASE_DIR / "transcript",
        "summary": BASE_DIR / "summary"
    }
    
    def __init__(self, base_dir: Path = None):
        if base_dir:
            self.BASE_DIR = Path(base_dir)
            self.PATHS = {
                key: self.BASE_DIR / path.name 
                for key, path in self.PATHS.items()
            }
            
        self.create_directories()
        self.SEGMENTATION_N = 1.5 
        self.SEGMENTATION_M = 10
        
        self.SCORING_K = 2.0
        self.SCORING_B = 0.75
        self.DOMINANT_PAIR_COUNT = 30
        self.DOMINANT_PAIR_BOOST = 1.2  
        
        self.TEMP_DIR = self.BASE_DIR / "temp_skims"
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.SUMMARY_DIR = self.BASE_DIR / "summaries"
        self.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
        
        self.WHISPER_MODEL_NAME = "base"  # Default model name for Whisper
        
    
    def create_directories(self):
        try:
            for path_name, path in self.PATHS.items():
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Directory {path} created or already exists.")
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            raise
    
    @property
    def video_upload_path(self) -> Path:
        return self.PATHS["video"]
    
    @property 
    def audio_path(self) -> Path:
        return self.PATHS["audio"]
    
    @property
    def transcript_path(self) -> Path:
        return self.PATHS["transcript"]
    
    @property
    def summary_path(self) -> Path:
        return self.PATHS["summary"]

# Get the project root directory (parent of the app directory)
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = project_root / "data"

# Initialize config with the data directory at project root level
config_settings = Config(base_dir=data_dir)

def get_config() -> Config:
    return config_settings