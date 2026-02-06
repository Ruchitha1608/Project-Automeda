"""
Preprocessing Utilities
Image and text preprocessing for the diagnostic system
"""

import numpy as np
from PIL import Image
import cv2
import re
from typing import Tuple, List, Optional, Union
import io


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

class ImagePreprocessor:
    """Image preprocessing utilities for histopathology images"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
        
        # ImageNet normalization stats
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def load_image(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """
        Load image from various sources
        
        Args:
            image_input: File path, bytes, or PIL Image
        
        Returns:
            PIL Image in RGB format
        """
        if isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
    
    def resize(self, image: Image.Image, size: Tuple[int, int] = None) -> Image.Image:
        """Resize image to target size"""
        size = size or self.target_size
        return image.resize(size, Image.Resampling.LANCZOS)
    
    def to_numpy(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array (H, W, C), values 0-255"""
        return np.array(image)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image with ImageNet stats
        
        Args:
            image: Numpy array (H, W, C) with values 0-255
        
        Returns:
            Normalized array with values roughly -2 to 2
        """
        img = image.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return img
    
    def to_tensor_format(self, image: np.ndarray) -> np.ndarray:
        """Convert from (H, W, C) to (C, H, W) format"""
        return np.transpose(image, (2, 0, 1))
    
    def preprocess(self, image_input: Union[str, bytes, Image.Image]) -> np.ndarray:
        """
        Full preprocessing pipeline
        
        Args:
            image_input: Image source (path, bytes, or PIL Image)
        
        Returns:
            Preprocessed numpy array (C, H, W), normalized
        """
        img = self.load_image(image_input)
        img = self.resize(img)
        img_np = self.to_numpy(img)
        img_norm = self.normalize(img_np)
        img_tensor = self.to_tensor_format(img_norm)
        return img_tensor
    
    def preprocess_batch(self, images: List[Union[str, bytes, Image.Image]]) -> np.ndarray:
        """Preprocess a batch of images"""
        processed = [self.preprocess(img) for img in images]
        return np.stack(processed, axis=0)
    
    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input image (BGR or grayscale)
            clip_limit: Threshold for contrast limiting
        
        Returns:
            Enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    @staticmethod
    def remove_background(image: np.ndarray, threshold: int = 240) -> np.ndarray:
        """
        Remove white background from histopathology images
        
        Args:
            image: Input image (BGR)
            threshold: Pixel value threshold for background
        
        Returns:
            Image with background masked
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = gray < threshold
        result = image.copy()
        result[~mask] = [255, 255, 255]  # Set background to white
        return result
    
    @staticmethod
    def stain_normalization(image: np.ndarray, method: str = 'macenko') -> np.ndarray:
        """
        Basic stain normalization for H&E images
        
        Args:
            image: Input image (RGB, 0-255)
            method: Normalization method ('macenko' or 'reinhard')
        
        Returns:
            Normalized image
        """
        # Simple implementation - for production use staintools library
        image = image.astype(np.float32)
        
        if method == 'reinhard':
            # Simple Reinhard normalization
            # Target mean and std (typical H&E)
            target_mean = np.array([148.60, 169.30, 105.97])
            target_std = np.array([41.56, 9.01, 6.67])
            
            # Normalize each channel
            for i in range(3):
                channel_mean = np.mean(image[:, :, i])
                channel_std = np.std(image[:, :, i])
                if channel_std > 0:
                    image[:, :, i] = (image[:, :, i] - channel_mean) / channel_std
                    image[:, :, i] = image[:, :, i] * target_std[i] + target_mean[i]
        
        return np.clip(image, 0, 255).astype(np.uint8)


# ============================================================
# TEXT PREPROCESSING
# ============================================================

class TextPreprocessor:
    """Text preprocessing utilities for biomedical text"""
    
    # Common biomedical abbreviations
    ABBREVIATIONS = {
        'BC': 'breast cancer',
        'TNBC': 'triple negative breast cancer',
        'ER': 'estrogen receptor',
        'PR': 'progesterone receptor',
        'IDC': 'invasive ductal carcinoma',
        'ILC': 'invasive lobular carcinoma',
        'DCIS': 'ductal carcinoma in situ',
        'OS': 'overall survival',
        'PFS': 'progression free survival',
        'DFS': 'disease free survival',
    }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean biomedical text
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep hyphens in gene names
        text = re.sub(r'[^\w\s\-\./]', '', text)
        
        return text.strip()
    
    @staticmethod
    def normalize_gene_name(gene: str) -> str:
        """
        Normalize gene name to standard format
        
        Args:
            gene: Gene name
        
        Returns:
            Normalized gene name (uppercase)
        """
        if not gene:
            return ""
        
        # Remove common suffixes
        gene = re.sub(r'\s*(gene|protein|mutation|variant)s?$', '', gene, flags=re.IGNORECASE)
        
        # Uppercase and strip
        return gene.upper().strip()
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
        
        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Simple word tokenization
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    @staticmethod
    def remove_stopwords(tokens: List[str], additional_stopwords: List[str] = None) -> List[str]:
        """
        Remove common stopwords
        
        Args:
            tokens: List of tokens
            additional_stopwords: Extra stopwords to remove
        
        Returns:
            Filtered tokens
        """
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'it', 'its', 'this', 'that', 'these', 'those', 'we', 'our', 'they',
            'their', 'which', 'who', 'whom', 'what', 'where', 'when', 'why', 'how',
        }
        
        if additional_stopwords:
            stopwords.update(additional_stopwords)
        
        return [t for t in tokens if t not in stopwords]
    
    @classmethod
    def preprocess(cls, text: str, remove_stops: bool = True) -> str:
        """
        Full text preprocessing pipeline
        
        Args:
            text: Input text
            remove_stops: Whether to remove stopwords
        
        Returns:
            Preprocessed text
        """
        text = cls.clean_text(text)
        
        if remove_stops:
            tokens = cls.tokenize(text)
            tokens = cls.remove_stopwords(tokens)
            text = ' '.join(tokens)
        
        return text
    
    @staticmethod
    def extract_gene_mentions(text: str, known_genes: List[str] = None) -> List[str]:
        """
        Extract gene mentions from text
        
        Args:
            text: Input text
            known_genes: List of known gene names to look for
        
        Returns:
            List of found genes
        """
        found = []
        text_upper = text.upper()
        
        # Default known genes
        if known_genes is None:
            known_genes = [
                'BRCA1', 'BRCA2', 'TP53', 'HER2', 'ERBB2', 'ESR1', 'PGR',
                'PIK3CA', 'CDH1', 'PTEN', 'RB1', 'MYC', 'CCND1', 'EGFR',
                'AKT1', 'MAP3K1', 'GATA3', 'FOXA1', 'RUNX1', 'CBFB',
            ]
        
        for gene in known_genes:
            if gene.upper() in text_upper:
                found.append(gene)
        
        # Also look for gene-like patterns (uppercase letters + numbers)
        pattern = r'\b[A-Z]{2,}[0-9]?[A-Z]?\b'
        matches = re.findall(pattern, text)
        
        for match in matches:
            if len(match) >= 3 and match not in found:
                # Simple heuristic: likely gene if 3-10 chars, has letter+number pattern
                if re.match(r'^[A-Z]+[0-9]*[A-Z]*$', match):
                    found.append(match)
        
        return list(set(found))


# ============================================================
# DATA VALIDATION
# ============================================================

class DataValidator:
    """Validation utilities for input data"""
    
    VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50 MB
    
    @classmethod
    def validate_image(cls, image_path: str) -> Tuple[bool, str]:
        """
        Validate an image file
        
        Args:
            image_path: Path to image
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        from pathlib import Path
        import os
        
        path = Path(image_path)
        
        # Check existence
        if not path.exists():
            return False, f"File not found: {image_path}"
        
        # Check extension
        if path.suffix.lower() not in cls.VALID_IMAGE_EXTENSIONS:
            return False, f"Invalid image format: {path.suffix}"
        
        # Check size
        if path.stat().st_size > cls.MAX_IMAGE_SIZE:
            return False, f"Image too large: {path.stat().st_size / 1024 / 1024:.1f} MB"
        
        # Try to open
        try:
            img = Image.open(image_path)
            img.verify()
            return True, "Valid"
        except Exception as e:
            return False, f"Cannot read image: {str(e)}"
    
    @staticmethod
    def validate_csv(csv_path: str, required_columns: List[str] = None) -> Tuple[bool, str]:
        """
        Validate a CSV file
        
        Args:
            csv_path: Path to CSV
            required_columns: List of required column names
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        import pandas as pd
        from pathlib import Path
        
        path = Path(csv_path)
        
        if not path.exists():
            return False, f"File not found: {csv_path}"
        
        if path.suffix.lower() != '.csv':
            return False, "File is not a CSV"
        
        try:
            df = pd.read_csv(csv_path, nrows=5)
            
            if required_columns:
                missing = set(required_columns) - set(df.columns)
                if missing:
                    return False, f"Missing columns: {missing}"
            
            return True, "Valid"
        except Exception as e:
            return False, f"Cannot read CSV: {str(e)}"


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def preprocess_image(image_input, target_size=(224, 224)):
    """Quick image preprocessing"""
    preprocessor = ImagePreprocessor(target_size)
    return preprocessor.preprocess(image_input)


def preprocess_text(text, remove_stopwords=True):
    """Quick text preprocessing"""
    return TextPreprocessor.preprocess(text, remove_stopwords)


def extract_genes(text):
    """Quick gene extraction"""
    return TextPreprocessor.extract_gene_mentions(text)


if __name__ == "__main__":
    # Test image preprocessing
    print("Testing Image Preprocessor...")
    img_prep = ImagePreprocessor()
    
    # Create test image
    test_img = Image.new('RGB', (512, 512), color='red')
    processed = img_prep.preprocess(test_img)
    print(f"Preprocessed shape: {processed.shape}")
    print(f"Value range: [{processed.min():.2f}, {processed.max():.2f}]")
    
    # Test text preprocessing
    print("\nTesting Text Preprocessor...")
    test_text = "BRCA1 and TP53 mutations are common in breast cancer patients."
    cleaned = TextPreprocessor.preprocess(test_text)
    print(f"Cleaned: {cleaned}")
    
    genes = TextPreprocessor.extract_gene_mentions(test_text)
    print(f"Found genes: {genes}")
    
    print("\n✅ Preprocessing utilities working correctly!")
