import os

# Ensure tests don't try to hit network
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-fake")
os.environ.setdefault("OPENAI_API_KEY", "test-key-fake")
os.environ.setdefault("AGENT_DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
