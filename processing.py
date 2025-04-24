import os
import sys
import logging
import datetime
from ui.core import Audio2MDUI # Import the UI class

def determine_file_paths(input_path: str,
                         output_suffix: str,
                         allowed_extensions: tuple[str] | str,
                         ui: Audio2MDUI, # Pass the UI instance
                         timestamp_format: str = "%Y%m%d_%H%M",
                         output_extension: str | None = None) -> tuple[list[tuple[str, str]], str]:
    """
    Determines input and output file paths for processing based on input path type.

    Handles both single file and directory inputs, validates extensions,
    and generates appropriate output paths with a timestamped suffix. Uses the
    provided Audio2MDUI instance for console output.

    Args:
        input_path (str): The path provided by the user (file or directory).
        output_suffix (str): Suffix to add to the output file/folder name (e.g., "_transcripts", "_orthography").
        allowed_extensions (tuple[str] | str): A tuple or single string of allowed lowercase file extensions (e.g., ".md", (".mp3", ".wav")).
        ui (Audio2MDUI): The Audio2MDUI instance for accessing the console.
        timestamp_format (str): Format string for the timestamp used in output names.
        output_extension (str | None, optional): If provided, forces the output file extension. Defaults to None (uses original extension).

    Returns:
        tuple[list[tuple[str, str]], str]:
            - A list of tuples, where each tuple contains (input_file_path, output_file_path).
            - A summary string describing the output location.

    Raises:
        SystemExit: If input path is invalid, extensions are wrong, or directories cannot be created.
    """
    console = ui.get_console() # Get console from the UI instance
    files_to_process = []
    output_location_summary = ""
    timestamp = datetime.datetime.now().strftime(timestamp_format)
    logging.debug(f"Generated timestamp: {timestamp}")

    # Ensure allowed_extensions is a tuple of lowercase strings
    if isinstance(allowed_extensions, str):
        allowed_extensions = (allowed_extensions.lower(),)
    else:
        allowed_extensions = tuple(ext.lower() for ext in allowed_extensions)

    if os.path.isfile(input_path):
        # --- Single File Input ---
        if not input_path.lower().endswith(allowed_extensions):
            ext_str = ', '.join(allowed_extensions)
            logging.critical(f"Input file does not have an allowed extension ({ext_str}): {input_path}")
            console.print(f"[error]Input file must have one of the following extensions: {ext_str}[/]", style="error")
            console.print(f"  [error]Provided file: {input_path}[/]", style="error")
            sys.exit(1)

        # Single file logic from original a2md_orthography.py
        base_name, original_ext = os.path.splitext(input_path)
        # Determine the final extension
        final_ext = output_extension if output_extension else original_ext
        # Construct the output filename directly, matching the original logic pattern
        output_file = f"{base_name}{output_suffix}_{timestamp}{final_ext}"

        files_to_process.append((input_path, output_file))
        output_location_summary = f"Output will be saved to: {output_file}"
        logging.info(f"Processing single file: {input_path} -> {output_file}")

    elif os.path.isdir(input_path):
        # --- Directory Input ---
        output_folder = input_path.rstrip("/\\") + f"{output_suffix}_{timestamp}"
        try:
            os.makedirs(output_folder, exist_ok=True)
            output_location_summary = f"Output will be saved to folder: {output_folder}"
            logging.info(f"Processing directory: {input_path} -> {output_folder}")
            # Use console for user-facing messages
            console.print(f"[info]Processing files from:[/info] {input_path}")
            console.print(f"[info]Saving output to:[/info]    {output_folder}")
            logging.info(f"Looking for files with extensions: {', '.join(allowed_extensions)}")
        except Exception as e:
            logging.critical(f"Could not create output directory {output_folder}: {e}", exc_info=True)
            console.print(f"[error]Could not create output directory {output_folder}: {e}[/]", style="error")
            sys.exit(1)

        found_files = False
        for file in sorted(os.listdir(input_path)):
            if file.lower().endswith(allowed_extensions):
                input_file = os.path.join(input_path, file)
                # Ensure it's a file, not a directory ending in the allowed extension
                if os.path.isfile(input_file):
                    # Directory logic: Output file keeps the *original* name inside the new folder
                    # BUT, we need to ensure the output file in the list has the correct *intended* extension if overridden
                    original_file_base, original_file_ext = os.path.splitext(file)
                    final_output_ext = output_extension if output_extension else original_file_ext
                    # The actual file saved might keep its name, but the path tracking needs the target extension
                    output_filename_in_folder = f"{original_file_base}{final_output_ext}"
                    output_file_path = os.path.join(output_folder, output_filename_in_folder)

                    output_file_path_in_folder = os.path.join(output_folder, file)
                    target_output_filename = f"{original_file_base}{final_output_ext}"
                    target_output_path = os.path.join(output_folder, target_output_filename)

                    files_to_process.append((input_file, target_output_path))

                    found_files = True
                    logging.debug(f"Adding file to process list: {input_file} -> {target_output_path}")
                else:
                    logging.debug(f"Skipping directory entry found with allowed extension: {input_file}")

        if not found_files:
             ext_str = ', '.join(allowed_extensions)
             logging.warning(f"No files with extensions ({ext_str}) found in the top-level of directory: {input_path}")
             console.print(f"[warning]No files with extensions ({ext_str}) found in the top-level of directory: {input_path}[/]", style="warning")
             sys.exit(0) # Exit gracefully if no files found

    else:
        # Invalid Input Path
        logging.critical(f"Input path not found or invalid: {input_path}")
        console.print(f"[error]Input path not found or invalid: {input_path}[/]", style="error")
        sys.exit(1)

    # Final check if list is empty (should be caught earlier, but safeguard)
    if not files_to_process:
        logging.warning("No files found to process after path evaluation.")
        console.print("[warning]No files found to process.[/]", style="warning")
        sys.exit(0)

    logging.info(f"Found {len(files_to_process)} file(s) to process.")
    console.print(f"\n[info]Found {len(files_to_process)} file(s) to process.[/]", style="info") # User feedback
    return files_to_process, output_location_summary