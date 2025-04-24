# llm_api_client.py
import logging
from typing import Optional
from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError, BadRequestError, APITimeoutError, APIError

# Import the config class itself, not an instance
try:
    from config import TranscriptionConfig
except ImportError:
    # Fallback or error handling if config structure changes
    logging.critical("Could not import TranscriptionConfig from config.py. Ensure it exists and contains LLM settings.")
    # Define a dummy TranscriptionConfig class to prevent NameError during import, but client will fail.
    class TranscriptionConfig:
        LLM_API_BASE_URL = None
        LLM_API_KEY = None
        LLM_TIMEOUT = 30
        LLM_RETRIES = 2
        LLM_MODEL_NAME = "default-model"
        LLM_TEMPERATURE = None

class LLMApiClient:
    """Handles initialization and interaction with the LLM API."""

    def __init__(self, config: TranscriptionConfig):
        """
        Initializes the client upon instantiation using a provided config instance.

        Args:
            config (TranscriptionConfig): An instance of the configuration class.
        """
        self.client: Optional[OpenAI] = None
        self.config = config # Store the passed config instance
        self._initialize_client()

    def _initialize_client(self):
        """Initializes the OpenAI client instance using settings from the stored config instance."""
        if self.client:
            logging.debug("LLM client already initialized.")
            return
        try:
            # Check if essential config values are present on the instance
            if not self.config.LLM_API_BASE_URL:
                 logging.error("LLM_API_BASE_URL is not configured. Cannot initialize LLM client.")
                 self.client = None
                 return

            logging.info("Initializing LLM API Client...")
            self.client = OpenAI(
                base_url=self.config.LLM_API_BASE_URL,
                api_key=self.config.LLM_API_KEY if self.config.LLM_API_KEY else "nokey", # Use dummy key if not provided
                timeout=self.config.LLM_TIMEOUT,
                max_retries=self.config.LLM_RETRIES,
            )
            logging.info(f"LLM Client initialized for model '{self.config.LLM_MODEL_NAME}' at {self.config.LLM_API_BASE_URL}")
        except Exception as e:
            # Log the specific error message along with the traceback
            logging.error(f"Failed to initialize OpenAI client. Error: {str(e)}", exc_info=True)
            self.client = None # Ensure client is None if init fails

    def is_available(self) -> bool:
        """Checks if the client was initialized successfully."""
        return self.client is not None

    def warm_up(self) -> bool:
        """
        Verifies LLM service connectivity and authentication.

        Returns:
            bool: True if the connection is successful, False otherwise.
                  Relies on logging for detailed error information.
        """
        if not self.is_available():
            logging.error("LLM Client not available. Skipping warm-up.")
            return False

        logging.info(f"Warming up LLM ({self.config.LLM_MODEL_NAME})...")
        try:
            # Use a lightweight call like listing models to check connection/auth
            self.client.models.list()
            logging.info("LLM connection check successful.")
            return True
        except AuthenticationError as e:
            logging.error(f"LLM warm-up failed: Authentication Error: {e}")
            return False
        except APIConnectionError as e:
            logging.error(f"LLM warm-up failed: Connection Error: {e}")
            return False
        except APIError as e: # Catch other specific API errors
            logging.error(f"LLM warm-up failed: API Error: {e}")
            return False
        except Exception as e: # Catch unexpected errors
            logging.error(f"LLM warm-up failed: Unexpected error: {e}", exc_info=True)
            return False

    def process_text(self, prompt: str) -> Optional[str]:
        """
        Processes the given prompt using the configured LLM.

        Args:
            prompt (str): The complete prompt to send to the LLM.

        Returns:
            Optional[str]: The processed text from the LLM, or None if an error occurred
                           or the LLM returned an empty response. Relies on logging
                           for detailed error information.
        """
        if not self.is_available():
            logging.error("LLM Client not available. Cannot process prompt.")
            return None

        logging.debug(f"Sending prompt to LLM (model: {self.config.LLM_MODEL_NAME}, temp: {self.config.LLM_TEMPERATURE}). Prompt start: {prompt[:100]}...")
        messages = [{"role": "user", "content": prompt}]

        try:
            # Build arguments, only including temperature if it's set in the config instance
            api_args = {
                "model": self.config.LLM_MODEL_NAME,
                "messages": messages,
            }
            if self.config.LLM_TEMPERATURE is not None:
                api_args["temperature"] = self.config.LLM_TEMPERATURE
                logging.debug(f"Using temperature from config: {self.config.LLM_TEMPERATURE}")
            else:
                 logging.debug("LLM_TEMPERATURE not set in config, using LLM service default.")

            response = self.client.chat.completions.create(**api_args)
            logging.debug(f"Raw LLM API response: {response}") # Log raw response if debugging

            # Check if choices array exists and has content
            if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
                 logging.warning("LLM response structure invalid or message content is missing/empty.")
                 return None # Treat malformed/empty response as failure

            refined_text = response.choices[0].message.content.strip()

            if refined_text:
                logging.info("LLM refinement successful.")
                return refined_text
            else:
                # This case might be redundant due to the check above, but kept for safety
                logging.warning("LLM returned an empty response after stripping.")
                return None # Treat empty response as failure for consistency

        except AuthenticationError as e:
            logging.error(f"LLM API call failed: Authentication Error: {e}")
            return None
        except APIConnectionError as e:
            logging.error(f"LLM API call failed: Connection Error after retries: {e}")
            return None
        except RateLimitError as e:
            logging.error(f"LLM API call failed: Rate Limit Error: {e}")
            return None
        except BadRequestError as e:
            logging.error(f"LLM API call failed: Bad Request Error: {e}")
            return None
        except APITimeoutError as e:
            logging.error(f"LLM API call failed: Timeout Error after retries: {e}")
            return None
        except APIError as e: # Catch other specific API errors
            logging.error(f"LLM API call failed: API Error: {e}")
            return None
        except Exception as e: # Catch unexpected errors
            logging.error(f"Unexpected error processing with LLM: {e}", exc_info=True)
            return None
