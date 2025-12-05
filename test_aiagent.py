"""
Unit tests for AIAgent class, focusing on chimera model functionality.

These tests verify that the chimera model is properly set up to use OpenRouter
and expose the issue where "chimera" is passed directly as the model name
instead of a valid OpenRouter model identifier.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys


# Mock streamlit before importing aiagent
@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock streamlit to avoid import issues in tests"""
    mock_st = MagicMock()
    mock_st.secrets = {}
    with patch.dict('sys.modules', {'streamlit': mock_st}):
        yield mock_st


@pytest.fixture
def mock_env_vars():
    """Set up mock environment variables for API keys"""
    env_vars = {
        "GOOGLE_API_KEY": "fake-google-key",
        "OPENAI_API_KEY": "fake-openai-key",
        "OPEN_ROUTER_API_KEY": "fake-openrouter-key",
        "ANTHROPIC_API_KEY": "fake-anthropic-key",
        "TOGETHER_API_KEY": "fake-together-key",
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_genai():
    """Mock Google generative AI"""
    with patch('aiagent.genai') as mock:
        mock.configure = Mock()
        mock.GenerativeModel = Mock()
        yield mock


@pytest.fixture
def mock_openai():
    """Mock OpenAI client"""
    with patch('aiagent.openai') as mock:
        mock_client = Mock()
        mock.OpenAI = Mock(return_value=mock_client)
        yield mock


class TestChimeraModelSetup:
    """Tests for chimera model setup and configuration"""
    
    def test_set_model_chimera_uses_openrouter_base_url(self, mock_env_vars, mock_genai, mock_openai):
        """Test that chimera model uses OpenRouter API endpoint"""
        from aiagent import AIAgent
        
        agent = AIAgent(model="chimera")
        
        # Verify OpenAI client was called with OpenRouter base URL
        mock_openai.OpenAI.assert_called()
        call_args = mock_openai.OpenAI.call_args
        assert call_args.kwargs.get('base_url') == "https://openrouter.ai/api/v1"
        assert call_args.kwargs.get('api_key') == "fake-openrouter-key"
    
    def test_set_model_chimera_stores_model_name(self, mock_env_vars, mock_genai, mock_openai):
        """Test that chimera model name is stored"""
        from aiagent import AIAgent
        
        agent = AIAgent(model="chimera")
        
        assert agent.model == "chimera"
    
    def test_set_summary_model_chimera_uses_openrouter(self, mock_env_vars, mock_genai, mock_openai):
        """Test that chimera summary model uses OpenRouter API"""
        from aiagent import AIAgent
        
        agent = AIAgent(model="gpt-4o-mini")
        agent.set_summary_model("chimera")
        
        # Verify OpenAI client was created with OpenRouter base URL
        calls = mock_openai.OpenAI.call_args_list
        openrouter_calls = [c for c in calls if c.kwargs.get('base_url') == "https://openrouter.ai/api/v1"]
        assert len(openrouter_calls) > 0


class TestChimeraModelQuery:
    """Tests for chimera model query functionality - exposes the silent failure bug"""
    
    def test_query_with_chimera_passes_model_name_to_api(self, mock_env_vars, mock_genai, mock_openai):
        """
        Test that documents the bug: 'chimera' is passed directly to OpenRouter API.
        
        OpenRouter expects actual model identifiers (e.g., 'openai/gpt-4', 
        'anthropic/claude-3-haiku'), not 'chimera'. This causes silent failures.
        """
        from aiagent import AIAgent
        
        # Setup mock response
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Test response"
        mock_completion.usage = Mock()
        mock_completion.usage.prompt_tokens = 100
        mock_completion.usage.completion_tokens = 50
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.OpenAI.return_value = mock_client
        
        # Mock moderation API
        mock_moderation = Mock()
        mock_moderation.results = [Mock()]
        mock_moderation.results[0].flagged = False
        mock_client.moderations.create.return_value = mock_moderation
        
        agent = AIAgent(model="chimera")
        agent.query("Hello")
        
        # Get what model name was passed to the API
        create_call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        model_passed = create_call_kwargs.get('model')
        
        # This documents the bug: "chimera" is passed as model name
        # OpenRouter expects real model identifiers like "openai/gpt-4"
        assert model_passed == "chimera", (
            "Expected 'chimera' to be passed to API (documenting the bug). "
            "When this test fails with a different model name, the bug is fixed."
        )
    
    def test_query_with_chimera_uses_chat_completions_api(self, mock_env_vars, mock_genai, mock_openai):
        """Test that chimera uses the OpenAI-compatible chat completions API"""
        from aiagent import AIAgent
        
        # Setup mock
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Test response"
        mock_completion.usage = Mock()
        mock_completion.usage.prompt_tokens = 100
        mock_completion.usage.completion_tokens = 50
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.OpenAI.return_value = mock_client
        
        mock_moderation = Mock()
        mock_moderation.results = [Mock()]
        mock_moderation.results[0].flagged = False
        mock_client.moderations.create.return_value = mock_moderation
        
        agent = AIAgent(model="chimera")
        response = agent.query("Hello")
        
        # Verify chat.completions.create was called
        mock_client.chat.completions.create.assert_called_once()
        assert response == "Test response"
    
    def test_query_chimera_includes_required_parameters(self, mock_env_vars, mock_genai, mock_openai):
        """Test that chimera query includes all required parameters"""
        from aiagent import AIAgent
        
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Test response"
        mock_completion.usage = Mock()
        mock_completion.usage.prompt_tokens = 100
        mock_completion.usage.completion_tokens = 50
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.OpenAI.return_value = mock_client
        
        mock_moderation = Mock()
        mock_moderation.results = [Mock()]
        mock_moderation.results[0].flagged = False
        mock_client.moderations.create.return_value = mock_moderation
        
        agent = AIAgent(model="chimera")
        agent.query("Hello", temperature=0.5, max_tokens=100)
        
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        
        assert 'model' in call_kwargs
        assert 'messages' in call_kwargs
        assert call_kwargs['temperature'] == 0.5
        assert call_kwargs['max_tokens'] == 100


class TestChimeraModelErrorHandling:
    """Tests for error handling when chimera model fails"""
    
    def test_chimera_invalid_model_error(self, mock_env_vars, mock_genai, mock_openai):
        """Test behavior when OpenRouter returns an invalid model error"""
        from aiagent import AIAgent
        
        # Simulate OpenRouter error for invalid model
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception(
            "Error: The model `chimera` does not exist"
        )
        mock_openai.OpenAI.return_value = mock_client
        
        agent = AIAgent(model="chimera")
        
        with pytest.raises(Exception) as exc_info:
            agent.query("Hello")
        
        assert "chimera" in str(exc_info.value)


class TestModelSwitching:
    """Tests for switching between models including chimera"""
    
    def test_switch_to_chimera_updates_base_url(self, mock_env_vars, mock_genai, mock_openai):
        """Test that switching to chimera uses OpenRouter endpoint"""
        from aiagent import AIAgent
        
        agent = AIAgent(model="gpt-4o-mini")
        agent.set_model("chimera")
        
        # Find the call that sets up chimera
        calls = mock_openai.OpenAI.call_args_list
        last_call = calls[-1]
        assert last_call.kwargs.get('base_url') == "https://openrouter.ai/api/v1"
    
    def test_switch_from_chimera_to_gpt(self, mock_env_vars, mock_genai, mock_openai):
        """Test that switching from chimera to GPT works correctly"""
        from aiagent import AIAgent
        
        agent = AIAgent(model="chimera")
        agent.set_model("gpt-4o-mini")
        
        calls = mock_openai.OpenAI.call_args_list
        last_call = calls[-1]
        assert last_call.kwargs.get('base_url') == "https://api.openai.com/v1"


class TestApiKeyHandling:
    """Tests for API key handling with chimera model"""
    
    def test_chimera_uses_open_router_api_key(self, mock_genai, mock_openai):
        """Test that chimera model uses OPEN_ROUTER_API_KEY"""
        with patch.dict(os.environ, {
            "GOOGLE_API_KEY": "fake-google-key",
            "OPENAI_API_KEY": "fake-openai-key",
            "OPEN_ROUTER_API_KEY": "test-openrouter-key-123",
        }):
            from aiagent import AIAgent
            
            agent = AIAgent(model="chimera")
            
            calls = mock_openai.OpenAI.call_args_list
            chimera_calls = [c for c in calls if c.kwargs.get('base_url') == "https://openrouter.ai/api/v1"]
            
            assert len(chimera_calls) > 0
            assert chimera_calls[-1].kwargs.get('api_key') == "test-openrouter-key-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
