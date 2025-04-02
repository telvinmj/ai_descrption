from flask import Blueprint, jsonify, request
from services.uncertain_descriptions_service import UncertainDescriptionsService
import os

uncertain_descriptions_bp = Blueprint('uncertain_descriptions', __name__)
service = UncertainDescriptionsService(os.getenv('DATABASE_PATH', 'dbt.db'))

@uncertain_descriptions_bp.route('/api/uncertain-descriptions', methods=['GET'])
def get_uncertain_descriptions():
    """Get all uncertain descriptions."""
    try:
        descriptions = service.get_uncertain_descriptions()
        return jsonify(descriptions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@uncertain_descriptions_bp.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    """Get sample data for a specific column."""
    try:
        model_name = request.args.get('modelName')
        column_name = request.args.get('columnName')
        
        if not model_name or not column_name:
            return jsonify({"error": "modelName and columnName are required"}), 400
        
        sample_data = service.get_sample_data(model_name, column_name)
        return jsonify(sample_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@uncertain_descriptions_bp.route('/api/description-feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for a description."""
    try:
        data = request.json
        
        if not data or 'id' not in data or 'improvedDescription' not in data:
            return jsonify({"error": "id and improvedDescription are required"}), 400
        
        service.submit_feedback(
            description_id=data['id'],
            improved_description=data['improvedDescription'],
            feedback=data.get('feedback', '')
        )
        
        return jsonify({"message": "Feedback submitted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500 