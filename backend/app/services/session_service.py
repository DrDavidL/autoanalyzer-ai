"""
Session Service
Manages user sessions and data storage
"""

import json
import pickle
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd
import redis
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("services.session")
settings = get_settings()

class SessionService:
    """Service for managing user sessions and data"""
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(settings.redis_url)
        self.session_ttl = 3600 * 24  # 24 hours
        
    async def create_session(
        self, 
        session_id: str = None, 
        user_id: str = None, 
        session_name: str = None
    ) -> Dict[str, Any]:
        """Create a new session"""
        if not session_id:
            session_id = str(uuid.uuid4())
            
        expires_at = datetime.utcnow() + timedelta(seconds=self.session_ttl)
        
        session_info = {
            "session_id": session_id,
            "user_id": user_id or "anonymous",
            "session_name": session_name or f"Session {session_id[:8]}",
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat(),
            "last_accessed": datetime.utcnow().isoformat()
        }
        
        # Store session info
        session_key = f"session:{session_id}"
        self.redis_client.setex(
            session_key,
            self.session_ttl,
            json.dumps(session_info)
        )
        
        logger.info(f"Created session {session_id}")
        return session_info
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        session_key = f"session:{session_id}"
        session_data = self.redis_client.get(session_key)
        
        if not session_data:
            return None
            
        session_info = json.loads(session_data)
        
        # Update last accessed
        session_info["last_accessed"] = datetime.utcnow().isoformat()
        self.redis_client.setex(
            session_key,
            self.session_ttl,
            json.dumps(session_info)
        )
        
        return session_info
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all associated data"""
        try:
            # Delete session info
            session_key = f"session:{session_id}"
            
            # Delete all session-related keys
            keys_to_delete = []
            for key in self.redis_client.scan_iter(match=f"session:{session_id}*"):
                keys_to_delete.append(key)
            for key in self.redis_client.scan_iter(match=f"data:{session_id}*"):
                keys_to_delete.append(key)
            for key in self.redis_client.scan_iter(match=f"results:{session_id}*"):
                keys_to_delete.append(key)
                
            if keys_to_delete:
                self.redis_client.delete(*keys_to_delete)
                
            logger.info(f"Deleted session {session_id} and {len(keys_to_delete)} associated keys")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    async def store_session_data(
        self, 
        session_id: str, 
        key: str, 
        data: Any
    ) -> bool:
        """Store data in session"""
        try:
            data_key = f"data:{session_id}:{key}"
            
            # Serialize data based on type
            if isinstance(data, pd.DataFrame):
                serialized_data = pickle.dumps(data)
            else:
                serialized_data = json.dumps(data, default=str)
                
            self.redis_client.setex(
                data_key,
                self.session_ttl,
                serialized_data
            )
            
            logger.debug(f"Stored data for session {session_id}, key {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing session data: {e}")
            return False
    
    async def get_session_data(self, session_id: str, key: str = None) -> Optional[Any]:
        """Get data from session"""
        try:
            if key:
                # Get specific key
                data_key = f"data:{session_id}:{key}"
                data = self.redis_client.get(data_key)
                
                if not data:
                    return None
                    
                # Try to deserialize as pickle first (for DataFrames)
                try:
                    return pickle.loads(data)
                except:
                    # Fall back to JSON
                    return json.loads(data)
            else:
                # Get all session data
                pattern = f"data:{session_id}:*"
                session_data = {}
                
                for data_key in self.redis_client.scan_iter(match=pattern):
                    key_name = data_key.decode().split(":")[-1]
                    data = self.redis_client.get(data_key)
                    
                    if data:
                        try:
                            session_data[key_name] = pickle.loads(data)
                        except:
                            session_data[key_name] = json.loads(data)
                
                return session_data if session_data else None
                
        except Exception as e:
            logger.error(f"Error getting session data: {e}")
            return None
    
    async def store_results(
        self, 
        session_id: str, 
        result_id: str, 
        results: Dict[str, Any]
    ) -> bool:
        """Store analysis results"""
        try:
            result_key = f"results:{session_id}:{result_id}"
            
            # Add metadata
            results["stored_at"] = datetime.utcnow().isoformat()
            results["session_id"] = session_id
            results["result_id"] = result_id
            
            serialized_results = json.dumps(results, default=str)
            self.redis_client.setex(
                result_key,
                self.session_ttl,
                serialized_results
            )
            
            logger.info(f"Stored results for session {session_id}, result {result_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing results: {e}")
            return False
    
    async def get_results(
        self, 
        session_id: str, 
        result_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """Get analysis results"""
        try:
            if result_id:
                # Get specific result
                result_key = f"results:{session_id}:{result_id}"
                data = self.redis_client.get(result_key)
                
                if not data:
                    return None
                    
                return json.loads(data)
            else:
                # Get all results for session
                pattern = f"results:{session_id}:*"
                all_results = {}
                
                for result_key in self.redis_client.scan_iter(match=pattern):
                    result_id = result_key.decode().split(":")[-1]
                    data = self.redis_client.get(result_key)
                    
                    if data:
                        all_results[result_id] = json.loads(data)
                
                return all_results if all_results else None
                
        except Exception as e:
            logger.error(f"Error getting results: {e}")
            return None
    
    async def list_sessions(self, user_id: str = None) -> list:
        """List all sessions, optionally filtered by user"""
        try:
            sessions = []
            pattern = "session:*"
            
            for session_key in self.redis_client.scan_iter(match=pattern):
                session_data = self.redis_client.get(session_key)
                if session_data:
                    session_info = json.loads(session_data)
                    
                    # Filter by user if specified
                    if user_id and session_info.get("user_id") != user_id:
                        continue
                        
                    sessions.append(session_info)
            
            # Sort by creation time
            sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return sessions
            
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions (Redis TTL should handle this, but manual cleanup for safety)"""
        try:
            cleaned = 0
            current_time = datetime.utcnow()
            
            for session_key in self.redis_client.scan_iter(match="session:*"):
                session_data = self.redis_client.get(session_key)
                if session_data:
                    session_info = json.loads(session_data)
                    expires_at = datetime.fromisoformat(session_info["expires_at"])
                    
                    if current_time > expires_at:
                        session_id = session_info["session_id"]
                        await self.delete_session(session_id)
                        cleaned += 1
            
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired sessions")
                
            return cleaned
            
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
            return 0
