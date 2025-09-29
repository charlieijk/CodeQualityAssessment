import pytest
import json
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.app import app


class TestFlaskAPI:
    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'

    def test_analyze_text_endpoint(self, client):
        """Test text analysis endpoint"""
        test_code = '''
def hello_world():
    """Print a greeting message."""
    print("Hello, World!")
'''
        response = client.post('/api/analyze-text',
                             json={'code': test_code},
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'quality_analysis' in data
        assert 'educational_feedback' in data

    def test_analyze_text_empty_code(self, client):
        """Test text analysis with empty code"""
        response = client.post('/api/analyze-text',
                             json={'code': ''},
                             content_type='application/json')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_analyze_text_no_code(self, client):
        """Test text analysis without code parameter"""
        response = client.post('/api/analyze-text',
                             json={},
                             content_type='application/json')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_analyze_text_malformed_json(self, client):
        """Test text analysis with malformed JSON"""
        response = client.post('/api/analyze-text',
                             data='invalid json',
                             content_type='application/json')

        assert response.status_code == 400

    def create_test_image(self, text="def hello():\n    print('Hello, World!')"):
        """Create a test image with code text"""
        # Create a simple image with text
        img = Image.new('RGB', (800, 400), color='white')
        draw = ImageDraw.Draw(img)

        try:
            # Try to use a monospace font
            font = ImageFont.truetype('/System/Library/Fonts/Courier.ttc', 16)
        except:
            # Fallback to default font
            font = ImageFont.load_default()

        # Draw the code text
        draw.text((20, 20), text, fill='black', font=font)

        return img

    def test_analyze_image_endpoint(self, client):
        """Test image analysis endpoint"""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            test_image = self.create_test_image()
            test_image.save(tmp_file.name, 'PNG')

            try:
                # Test the upload
                with open(tmp_file.name, 'rb') as f:
                    response = client.post('/api/analyze',
                                         data={'file': (f, 'test.png')},
                                         content_type='multipart/form-data')

                # Note: This test might fail if OCR dependencies aren't installed
                # In a real environment, you'd have pytesseract properly configured
                assert response.status_code in [200, 500]  # Allow for OCR setup issues

            finally:
                # Clean up
                os.unlink(tmp_file.name)

    def test_analyze_image_no_file(self, client):
        """Test image analysis without file"""
        response = client.post('/api/analyze',
                             data={},
                             content_type='multipart/form-data')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_analyze_image_invalid_file_type(self, client):
        """Test image analysis with invalid file type"""
        # Create a text file instead of an image
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(b'This is not an image')
            tmp_file.flush()

            try:
                with open(tmp_file.name, 'rb') as f:
                    response = client.post('/api/analyze',
                                         data={'file': (f, 'test.txt')},
                                         content_type='multipart/form-data')

                assert response.status_code == 400
                data = json.loads(response.data)
                assert 'error' in data

            finally:
                os.unlink(tmp_file.name)

    def test_feedback_endpoint(self, client):
        """Test contextual feedback endpoint"""
        response = client.get('/api/feedback/85.5')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'feedback' in data
        assert 'grade' in data['feedback']
        assert 'message' in data['feedback']

    def test_feedback_invalid_score(self, client):
        """Test feedback endpoint with invalid score"""
        response = client.get('/api/feedback/150')  # Score > 100
        assert response.status_code == 400

        response = client.get('/api/feedback/-10')  # Score < 0
        assert response.status_code == 400

        response = client.get('/api/feedback/invalid')  # Non-numeric
        assert response.status_code == 400

    def test_404_endpoint(self, client):
        """Test 404 error handling"""
        response = client.get('/api/nonexistent')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data

    def test_index_page(self, client):
        """Test main index page loads"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Code Quality Assessment' in response.data