from __future__ import annotations  # allows using newer Python type hints in older Python versions

import os  # lets us read environment variables and interact with the operating system
from dotenv import load_dotenv  # reads the .env file and loads variables into the environment
import uvicorn  # uvicorn is the web server that runs our FastAPI app

load_dotenv()  # reads .env file RIGHT NOW before anything else — this is how GROQ_API_KEY etc. get loaded

if __name__ == "__main__":  # only runs this block if you directly run "python run.py" (not when imported)

    host = os.getenv("API_HOST", "0.0.0.0")  # reads API_HOST from .env; if not set, defaults to "0.0.0.0" (all network interfaces)
    port = int(os.getenv("API_PORT", "8020"))  # reads API_PORT from .env; if not set, defaults to 8020; convert to int because env vars are always strings

    uvicorn.run(
        "app.main:app",  # tells uvicorn: find the file app/main.py and use the variable called 'app' inside it
        host=host,       # which network address to listen on
        port=port,       # which port number to listen on (8020 by default)
        reload=True,     # auto-restart the server whenever you save a .py file (useful during development)
    )
