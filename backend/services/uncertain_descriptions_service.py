from typing import List, Dict, Any, Optional
import sqlite3
from datetime import datetime

class UncertainDescriptionsService:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_uncertain_descriptions(self) -> List[Dict[str, Any]]:
        """Get all descriptions with low confidence scores."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get columns with low confidence or marked as uncertain
            cursor.execute("""
                SELECT 
                    c.id,
                    m.name as model_name,
                    c.name as column_name,
                    c.description as current_description,
                    c.ai_confidence_score,
                    c.uncertainty_reason
                FROM columns c
                JOIN models m ON c.model_id = m.id
                WHERE c.ai_confidence_score < 0.7
                OR c.needs_review = 1
                ORDER BY c.ai_confidence_score ASC
            """)
            
            rows = cursor.fetchall()
            descriptions = []
            
            for row in rows:
                descriptions.append({
                    "id": str(row[0]),
                    "modelName": row[1],
                    "columnName": row[2],
                    "currentDescription": row[3] or "",
                    "confidenceScore": float(row[4]) if row[4] is not None else 0.0,
                    "uncertaintyReason": row[5] or "Low confidence score"
                })
            
            return descriptions
        finally:
            conn.close()

    def get_sample_data(self, model_name: str, column_name: str) -> List[Dict[str, Any]]:
        """Get sample data for a specific column."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get sample data from the sample_data table
            cursor.execute("""
                SELECT sd.value, COUNT(*) as count
                FROM sample_data sd
                JOIN columns c ON sd.column_id = c.id
                JOIN models m ON c.model_id = m.id
                WHERE m.name = ? AND c.name = ?
                GROUP BY sd.value
                ORDER BY count DESC
                LIMIT 10
            """, (model_name, column_name))
            
            rows = cursor.fetchall()
            return [{"value": row[0], "count": row[1]} for row in rows]
        finally:
            conn.close()

    def submit_feedback(self, description_id: str, improved_description: str, feedback: str) -> None:
        """Submit feedback for a description."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Update the description and mark it as user-edited
            cursor.execute("""
                UPDATE columns
                SET 
                    description = ?,
                    user_edited = 1,
                    needs_review = 0,
                    last_edited = ?
                WHERE id = ?
            """, (improved_description, datetime.now().isoformat(), description_id))
            
            # Store the feedback
            if feedback:
                cursor.execute("""
                    INSERT INTO description_feedback (
                        column_id,
                        feedback,
                        timestamp
                    ) VALUES (?, ?, ?)
                """, (description_id, feedback, datetime.now().isoformat()))
            
            conn.commit()
        finally:
            conn.close() 