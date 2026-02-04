import asyncio
import logging
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import redis
from redisvl.utils.vectorize import CustomTextVectorizer
from redisvl.extensions.cache.llm import SemanticCache
from utils.database import get_pg_connection, fetchall, fetch_one, get_redis_client, get_redis_checkpointer
from typing import Optional, List
from agent.state import UserEmotion
load_dotenv()