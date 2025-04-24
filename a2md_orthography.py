#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import datetime
from dotenv import load_dotenv
from openai import APIConnectionError, AuthenticationError, RateLimitError, BadRequestError, APITimeoutError, APIError
from rich.progress import Progress # Keep for type hinting
from rich.table import Table # Needed for summary table creation via ui.core
from rich.console import Console # Needed for type hinting
from ui.core import Audio2MDUI # Import the centralized UI class
from llm_api_client import LLMApiClient # Import the new consolidated client
from processing import determine_file_paths # Import the new function

# Load environment variables early - needed for API configuration
# and logging setup before any operations begin
load_dotenv()

# Import the configuration class
from config import TranscriptionConfig # Import class
Config = TranscriptionConfig() # Instantiate AFTER load_dotenv()

# Core processing functions
# Handle the main workflow of reading, refining and saving transcripts
def read_prompt_template(file_path: str, console: Console) -> str:
    """
    Read and return the contents of a prompt template file.

    This function reads the contents of a file specified by `file_path` and returns
    it as a string. It handles potential exceptions that may occur during file reading.

    Args:
        file_path (str): Path to the prompt template file to read.
        console (Console): Rich console instance for printing errors.

    Returns:
        str: The file contents as a string, or empty string on failure.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        UnicodeDecodeError: If file contains invalid UTF-8 characters.

    NB:
        Errors are logged and printed to the console before returning empty string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"Prompt template file not found: {file_path}")
        console.print(f"\n[error]Prompt template file not found: {file_path}[/]", style="error")
        return ""
    except Exception as e:
        logging.error(f"Error reading prompt template {file_path}: {e}", exc_info=True)
        console.print(f"\n[error]Could not read prompt template file: {e}[/]", style="error")
        return ""

def refine_transcript_text(llm_api: LLMApiClient, transcript_text: str, prompt_template: str, progress: Progress = None) -> str:
    """
    Orchestrate LLM processing of text chunks with overlap handling using LLMApiClient.

    This function coordinates the processing of the input transcript text by
    splitting it into overlapping chunks, refining each chunk using the LLM,
    deduplicating the results, and combining them into a final output.

    Args:
        llm_api (LLMApiClient): Initialized LLM API client instance.
        transcript_text (str): Full input text to process.
        prompt_template (str): Template string for LLM prompts.
        progress (Progress, optional): Rich progress tracker for visual feedback.
            Defaults to None.

    Returns:
        str: Combined and deduplicated refined text, or the original text if refinement fails.

    Processing Steps:
        1. Split text into overlapping chunks using `chunk_with_overlap`.
        2. Process each chunk through LLM using `llm_api.process_text`.
        3. Deduplicate overlapping sections using `trim_redundant_overlap`.
        4. Combine refined chunks into final output.
    """

    if not transcript_text or not prompt_template:
        logging.warning("Cannot refine empty transcript or without prompt template.")
        return "" # Return empty string if input is invalid

    # NEW: Overlapping chunks
    fragments = chunk_with_overlap(
        transcript_text,
        Config.FRAGMENT_SIZE,
        Config.CHUNK_OVERLAP
    )

    refined_fragments = []
    llm_failed_fragments = 0
    logging.info(f"Refining transcript ({len(transcript_text)} chars) in {len(fragments)} overlapping fragments...")

    task_id = progress.add_task("[cyan]Refining fragments...", total=len(fragments), visible=True, phase="Refining") if progress else None

    for i, fragment in enumerate(fragments):
        if progress:
            progress.update(task_id, advance=1, description=f"[cyan]Refining fragment {i+1}/{len(fragments)}")
        prompt = prompt_template.format(
            part_number=i + 1,
            total_parts=len(fragments),
            chunk=fragment
        )
        logging.debug(f"Processing fragment {i+1}/{len(fragments)}")
        # Use the new client method
        refined_fragment = llm_api.process_text(prompt)

        # Check for None return value from process_text (indicates failure or empty response)
        if refined_fragment is None:
            logging.warning(f"LLM refinement failed for fragment {i + 1}. Using original fragment.")
            refined_fragments.append(fragment) # Use original fragment on failure
            llm_failed_fragments += 1
        else:
            refined_fragments.append(refined_fragment)

    if llm_failed_fragments > 0:
        warning_msg = f"{llm_failed_fragments}/{len(fragments)} fragments failed refinement. Original text used for failed fragments."
        logging.warning(warning_msg)
        # Optionally print to console if needed: console.print(f"[warning]{warning_msg}[/]")

    # Deduplicate overlapping content even if some fragments failed
    if not refined_fragments: # Handle case where all fragments failed
         logging.error("All fragments failed refinement. Returning original text.")
         return transcript_text # Or return "" depending on desired behavior

    deduped_fragments = [refined_fragments[0]]
    for i in range(1, len(refined_fragments)):
        deduped = trim_redundant_overlap(deduped_fragments[-1], refined_fragments[i])
        deduped_fragments.append(deduped)

    final_text = "\n\n".join(deduped_fragments)
    # Return original text if the final joined text is empty after processing
    return final_text if final_text else transcript_text


def trim_redundant_overlap(a: str, b: str, min_overlap: int = 20) -> str:
    """
    Remove duplicate content between consecutive text chunks.

    This function examines the overlap between two consecutive text chunks and
    removes any redundant content from the second chunk that is identical to
    the end of the first chunk.

    Args:
        a (str): Previous chunk text.
        b (str): Current chunk text.
        min_overlap (int, optional): Minimum overlap to check for (characters).
            Defaults to 20.

    Returns:
        str: Current chunk with redundant prefix removed.

    Example:
        >>> trim_redundant_overlap("end of previous", "previous start")
        'start'
    """
    max_check = min(len(a), len(b), 200)  # Limit scan window
    for i in range(max_check, min_overlap - 1, -1):
        if a[-i:].strip() == b[:i].strip():
            return b[i:].lstrip()
    return b

def chunk_with_overlap(text: str, size: int, overlap: int) -> list[str]:
    """
    Split text into context-preserving chunks with overlap.

    This function uses a sliding window approach to divide the input text into
    overlapping segments. It ensures that adjacent chunks have the specified
    overlap, maintaining context between them.

    Args:
        text (str): Input text to be chunked.
        size (int): Maximum characters per chunk.
        overlap (int): Number of characters to overlap between adjacent chunks.
            Must be less than size.

    Returns:
        list[str]: List of text chunks with specified overlap.

    Raises:
        ValueError: If size <= 0, overlap < 0, or overlap >= size.

    Example:
        >>> chunk_with_overlap("abcdefghij", 4, 2)
        ['abcd', 'cdef', 'efgh', 'ghij']
    """
    if size <= 0:
        raise ValueError("Chunk size must be greater than 0")
    if overlap < 0:
        raise ValueError("Overlap must be non-negative")
    if overlap >= size:
        raise ValueError("Overlap must be smaller than chunk size")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= len(text):
            break
        start = end - overlap  # Move start back by overlap

    return chunks

def process_transcript_file(llm_api: LLMApiClient, input_file_path: str, output_file_path: str, prompt_template: str, progress: Progress = None) -> bool:
    """
    Execute full transcript processing pipeline using LLMApiClient.

    This function manages the complete workflow of processing a transcript file,
    from reading the input Markdown file to saving the refined output. It includes
    error handling and progress tracking using Rich.

    Args:
        llm_api (LLMApiClient): Initialized LLM API client instance.
        input_file_path (str): Path to source Markdown file to process.
        output_file_path (str): Destination path for refined text output.
        prompt_template (str): LLM prompt template string for refinement.
        progress (Progress, optional): Rich progress tracker for visual feedback.
            Defaults to None.

    Returns:
        bool: True if processing succeeded, False on failure.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        OSError: If output directory cannot be created.
        Exception: For other processing errors.

    Workflow:
        1. Validate input paths and contents.
        2. Read source file content.
        3. Refine text through LLM in chunks using `refine_transcript_text`.
        4. Create output directory if needed.
        5. Save refined text with error handling.
    """
    logging.info(f"Processing: {os.path.basename(input_file_path)}")
    # No print here, rely on logging

    try:
        # Read the raw transcript
        with open(input_file_path, 'r', encoding='utf-8') as f:
            raw_transcript_text = f.read()
        logging.debug(f"Read {len(raw_transcript_text)} chars from {input_file_path}")

        if not raw_transcript_text:
            logging.warning(f"Input transcript file is empty: {input_file_path}. Skipping.")
            # No print here, rely on logging
            return False # Consider this a skip

        # Refine the text using the passed LLM client
        refined_transcript = refine_transcript_text(llm_api, raw_transcript_text, prompt_template, progress)

        # Check for None return value (failure or empty) or if it reverted to original
        if refined_transcript is None or refined_transcript == raw_transcript_text:
            if refined_transcript is None:
                 logging.error(f"LLM refinement failed for {input_file_path}.")
            else:
                 # This case happens if refine_transcript_text returns original due to failures
                 logging.warning(f"LLM refinement failed for one or more chunks in {input_file_path}. Output file contains original text for failed chunks.")
            # Decide if partial failure should still write the file or count as failure
            # Current logic: If refine_transcript_text returns original, we still write it.
            # If it returns None (total failure), we mark as False.
            if refined_transcript is None:
                 return False # Indicate failure

        # Ensure output directory exists and save the refined transcript
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                logging.debug(f"Ensured output directory exists: {output_dir}")
            except Exception as e:
                logging.error(f"Could not create output directory {output_dir}: {e}", exc_info=True)
                # No print here, rely on logging
                return False # Cannot save file

        with open(output_file_path, 'w', encoding='utf-8') as md_file:
            md_file.write(refined_transcript)
        logging.info(f"Successfully saved REFINED transcript to {output_file_path}")
        # No print here, rely on logging
        return True

    except FileNotFoundError:
        logging.error(f"Input transcript file not found: {input_file_path}")
        # No print here, rely on logging
        return False
    except Exception as e:
        logging.error(f"Error during processing or saving for {input_file_path}: {e}", exc_info=True)
        # No print here, rely on logging
        return False

def run():
    """
    Main CLI entry point for transcript processing.

    This function handles the command-line interface, argument parsing, and validation.
    It orchestrates the complete processing workflow, including system configuration,
    logging setup, LLM service warm-up, file/directory processing, and Rich-based
    progress tracking and display.

    Args (via command line):
        input_path (str): Path to .md file or directory of files to process.
        --prompt-file (str, optional): Path to prompt template file. Defaults to 'prompts/transcript_prompt.txt'.

    Returns:
        None: Exits with status code 0 on success, 1 on error.

    Raises:
        SystemExit: For invalid arguments or critical failures.
    """
    # UI and Logging Setup
    ui = Audio2MDUI()
    console = ui.get_console() # Get the shared console instance
    # Create Live display components (Live object, Progress bar, LogPanel)
    live, progress, log_panel = ui.create_live_display()
    # Configure logging to use the LogPanel within the Live display
    script_base_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    ui.setup_live_logging(log_panel, Config.DEBUG, script_base_name)
    
    parser = argparse.ArgumentParser(description="Apply LLM orthography correction to Markdown transcript files.")
    parser.add_argument("input_path", help="Path to a single .md transcript file or a directory containing .md files.")
    parser.add_argument("--prompt-file", default="prompts/transcript_prompt.txt", help="Path to the LLM prompt template file (default: prompts/transcript_prompt.txt).")

    args = parser.parse_args()
    logging.debug(f"Parsed arguments: {args}")

    input_path = args.input_path
    prompt_file_path = args.prompt_file

    # Log initial config values from OrthographyConfig
    # LLM specific values are read by LLMApiClient from TranscriptionConfig
    logging.info(f"Fragment Size (Orthography): {Config.FRAGMENT_SIZE}")
    logging.info(f"Chunk Overlap (Orthography): {Config.CHUNK_OVERLAP}")
    logging.info(f"Debug Logging Enabled: {Config.DEBUG}")

    # Validate chunk overlap
    if Config.CHUNK_OVERLAP >= Config.FRAGMENT_SIZE:
        console.print(
            f"[error]CHUNK_OVERLAP ({Config.CHUNK_OVERLAP}) must be smaller than FRAGMENT_SIZE ({Config.FRAGMENT_SIZE}).[/]",
            style="error"
        )
        sys.exit(1)

    # Initial Checks
    logging.info("Initializing LLM API Client...")
    # Pass the instantiated config object to the client
    llm_api = LLMApiClient(config=Config)

    if not llm_api.is_available():
        console.print("[error]LLM Client failed to initialize. Check config and logs.[/]", style="error")
        logging.critical("LLM Client initialization failed. Exiting.")
        sys.exit(1)

    logging.info("Attempting to warm up LLM...")
    if not llm_api.warm_up():
        # Error should be logged by warm_up method in LLMApiClient
        console.print("[error]LLM warm-up failed. Check connection/authentication in logs.[/]", style="error")
        logging.critical("LLM warm-up failed. Exiting.")
        sys.exit(1)
    else:
        # Print success message here, as warm_up only returns bool
        console.print("[success]LLM service responded.[/]", style="success")

    prompt_template = read_prompt_template(prompt_file_path, console) # Pass console instance
    if not prompt_template:
        # Error logged and printed within read_prompt_template
        logging.critical(f"Cannot proceed without prompt template: {prompt_file_path}")
        sys.exit(1)
    logging.info(f"Loaded prompt template from {prompt_file_path}")

    # Determine input/output file paths using the refactored function
    # NB: determine_file_paths handles console output via the ui instance
    try:
        files_to_process, output_location_summary = determine_file_paths(
            input_path=input_path,
            output_suffix="_orthography", # Suffix for orthography files/folders
            allowed_extensions=".md", # Only process markdown files
            ui=ui, # Pass the initialized UI instance
            output_extension=".md" # Explicitly set output extension
        )
        # Output location summary is now handled within determine_file_paths via console.print
        # if output_location_summary:
        #     console.print(f"[info]{output_location_summary}[/]", style="info")

    except SystemExit:
        # determine_file_paths calls sys.exit on errors, so we just exit here too
        sys.exit(1)
    except Exception as e:
        # Catch any unexpected errors during path determination
        logging.critical(f"An unexpected error occurred during file path determination: {e}", exc_info=True)
        ui.get_console().print(f"[error]An unexpected error occurred: {e}[/]", style="error")
        sys.exit(1)

    # The checks for empty list and printing the count are now inside determine_file_paths
    # if not files_to_process:
    #     # Logging and printing handled within determine_file_paths
    #     sys.exit(0)
    # logging.info(f"Found {len(files_to_process)} transcript file(s) to process.") # Handled by determine_file_paths

    # Processing Loop with Rich Live Display
    success_count = 0
    failure_count = 0
    logging.info(f"Starting LLM refinement for {len(files_to_process)} files.")

    # Use the 'live' context manager returned by ui.create_live_display()
    with live:
        # Use the 'progress' instance returned by ui.create_live_display()
        overall_task = progress.add_task("[green]Overall Progress", total=len(files_to_process), phase="Overall")

        for input_f, output_f in files_to_process:
            # Pass the 'progress' instance to the processing function
            progress.update(overall_task, description=f"[green]Processing: {os.path.basename(input_f)}")
            # Pass the llm_api instance to the processing function
            if process_transcript_file(llm_api, input_f, output_f, prompt_template, progress):
                success_count += 1
            else:
                failure_count += 1
                # Logging already handled inside process_transcript_file
            progress.update(overall_task, advance=1) # Advance overall progress after file attempt

        # Stop the overall task
        progress.update(overall_task, description="[bold green]Overall Process Complete")
        progress.stop_task(overall_task)

        # Allow Live display to refresh one last time before printing summary
        live.refresh()

        # Final Summary
        summary_msg = f"Overall Summary: Successfully styled: {success_count} file(s), Failed or skipped: {failure_count} file(s)."
        logging.info(summary_msg)
        # Table creation is now handled by ui.summary_table()

        live.console.print("\n[info]--- LLM Refinement Complete ---[/]") # Print below live display

        # Prepare data for the summary table
        summary_data = {
            "Successfully styled": f"{success_count} file(s)",
            "Failed or skipped": f"{failure_count} file(s)"
        }
        if output_location_summary:
            summary_data["Output location"] = output_location_summary

        # Create the table using the UI helper method
        summary_table = ui.summary_table(data=summary_data, title="Refinement Summary")

        live.console.print(summary_table) # Print summary table below live display

if __name__ == "__main__":
    run()