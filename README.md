# Advanced Discord Analysis Bot

## ⚠️ Security Notice

**IMPORTANT**: This project requires sensitive credentials (API keys, database passwords, etc.).
- **NEVER** commit your `.env` file or any file containing real credentials to version control
- Always use environment variables for sensitive data
- Review the `.env.example` file for required configuration

## Overview

This repository contains the implementation of an advanced Discord bot designed to manage and analyze a repository of messages, particularly focusing on link unfurling and contextual understanding. The bot is equipped with AI-driven functionalities to process historical and real-time messages, transform links for enhanced readability, and query a knowledge base for information retrieval.

## System Architecture

The bot operates on a modular architecture comprising several Python scripts, each responsible for a distinct part of the functionality:

- `config.py`: Centralized configuration management.
- `db_manager.py`: Abstraction layer for all database transactions.
- `link_transformer.py`: Utility module for link conversion and metadata extraction.
- `knowledge_base.py`: Interface for the AI-enhanced knowledge base, supporting CRUD operations.
- `message_fetcher.py`: Service for historical message extraction from Discord channels.
- `search_handler.py`: The search engine component that handles complex query processing.
- `bot.py`: The core bot functionality integrating Discord events and commands.

## Key Components

- **Virtual Environment Setup**: Detailed instructions for setting up an isolated Python environment.
- **Dependency Management**: `requirements.txt` for managing library dependencies.
- **Data Model**: Schemas for the database reflecting the structured message and link metadata.
- **AI Integration**: Utilization of state-of-the-art NLP models for knowledge extraction and query handling.
- **Link Unfurling**: Implementation details on the transformation of links to enhance content preview.
- **Historical Data Processing**: Strategies for backfilling the knowledge base with historical server data.
- **Real-time Message Handling**: Event-driven architecture for processing incoming messages.
- **Knowledge Base Maintenance**: Procedures for updating and maintaining the knowledge base.
- **Search Optimization**: Vector space modeling and indexing for efficient search capabilities.

## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL database
- Elasticsearch instance
- Discord Bot Token
- OpenAI API Key

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd docspdfsnotebooks
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install python-dotenv  # Required for environment variable management
   ```

4. **Configure environment variables**

   Copy the example environment file and configure it with your credentials:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your credentials:
   ```bash
   # Discord Configuration
   DISCORD_TOKEN=your_discord_bot_token_here

   # Database Configuration
   DB_USER=postgres
   DB_PASSWORD=your_database_password_here
   DB_NAME=data_disco
   DB_HOST=localhost
   DB_PORT=5432

   # Elasticsearch Configuration
   ELASTICSEARCH_URL=https://localhost:9200
   ELASTICSEARCH_USER=elastic
   ELASTICSEARCH_PASSWORD=your_elasticsearch_password_here
   ELASTICSEARCH_VERIFY_CERTS=False

   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Set up the database**

   Create the PostgreSQL database and required tables:
   ```bash
   createdb data_disco
   # Run your database migration scripts here
   ```

6. **Run the bot**
   ```bash
   python bot.py
   ```

### Security Best Practices

- **Never commit** the `.env` file to version control
- Use strong, unique passwords for all services
- Rotate API keys and passwords regularly
- Limit database user permissions to only what's necessary
- Use SSL/TLS for database and Elasticsearch connections in production
- Review the `.gitignore` file to ensure sensitive files are excluded

## Usage

- **Starting the Bot**: Command-line instructions for running the bot.
- **Commands**: Documentation of the bot's commands and their usage.
- **Querying the Knowledge Base**: Examples of complex queries and how the bot processes and responds to them.

## Contributing

Guidelines for contributing to the project, including code style, pull request process, and issue tracking.

## License

Details of the project's license.

## Authors

Credits to the contributors and maintainers of the project.
