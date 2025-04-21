from pathlib import Path
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    # Default paths configuration
    BASE_DIR = Path("./data")
    PATHS = {
        "video": BASE_DIR / "video",
        "audio": BASE_DIR / "audio", 
        "transcript": BASE_DIR / "transcript"
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
        
        # Whisper configuration
        self.WHISPER_MODEL_NAME = "tiny"  # Default model name for Whisper
        
        # Azure Speech Service configuration
        self.AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
        self.AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")
        self.AZURE_SPEECH_ENDPOINT = os.getenv("AZURE_SPEECH_ENDPOINT", "https://eastus.api.cognitive.microsoft.com/")
        self.USE_AZURE_SPEECH = os.getenv("USE_AZURE_SPEECH", "False").lower() == "true"
        self.WHISPER_LOCAL = os.getenv("WHISPER_LOCAL", "False").lower() == "true"
    
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

# Get the project root directory (parent of the app directory)
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = project_root / "data"

# Initialize config with the data directory at project root level
config_settings = Config(base_dir=data_dir)

def get_config() -> Config:
    return config_settings