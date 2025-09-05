"""
Optimization utilities for the attendance system.
Provides caching and batch processing capabilities.
"""
import time
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class MemoryCache:
    """
    A thread-safe in-memory cache with LRU eviction policy.
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize the memory cache.
        
        Args:
            max_size (int): Maximum number of items in cache
            ttl (int): Time to live in seconds for cached items
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            The cached value or None if not found/expired
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check if item has expired
            if time.time() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            # Move to end (LRU)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in cache.
        
        Args:
            key (str): Cache key
            value (Any): Value to cache
        """
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
            
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            # Add new item
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)


class BatchProcessor:
    """
    A utility class for processing items in batches.
    """
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize the batch processor.
        
        Args:
            batch_size (int): Size of each batch
        """
        self.batch_size = batch_size
    
    def process_batches(self, items: List[Any], process_func) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items (List[Any]): List of items to process
            process_func: Function to apply to each batch
            
        Returns:
            List of processed results
        """
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = process_func(batch)
            results.extend(batch_results)
        return results
    
    def process_embeddings_batch(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Process a batch of embeddings.
        
        Args:
            embeddings (List[np.ndarray]): List of embedding arrays
            
        Returns:
            np.ndarray: Stacked embeddings
        """
        if not embeddings:
            return np.array([])
        return np.stack(embeddings)
    
    def split_into_batches(self, items: List[Any]) -> List[List[Any]]:
        """
        Split items into batches.
        
        Args:
            items (List[Any]): List of items to split
            
        Returns:
            List of batches
        """
        return [items[i:i + self.batch_size] 
                for i in range(0, len(items), self.batch_size)]
