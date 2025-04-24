"""
UI Core Module for Audio2MD
Provides consistent UI components across all scripts
"""

import logging
import sys
import os
import datetime
from collections import deque
from typing import Dict, Optional, Tuple
import traceback

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn
)
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.align import Align

class LogPanel:
    """
    Rich-compatible panel for capturing and displaying log messages.

    Maintains a scrollback buffer for display in Rich UI.
    Implements a stream-like interface for logging handlers.

    Attributes:
        max_lines (int): Maximum lines to retain in scrollback buffer.
    """
    def __init__(self, console, max_lines=20):
        self.console = console
        self._lines = deque(maxlen=max_lines)

    def write(self, text: str) -> None:
        """Add new log entry to the panel."""
        self._lines.append(text.strip())

    def __rich__(self) -> Text:
        """Render log panel content for Rich display."""
        lines_snapshot = list(self._lines)
        log_text = Text("\n".join(lines_snapshot))
        return log_text

class LogPanelStream:
    """Stream-like object to direct RichHandler output to LogPanel."""
    def __init__(self, log_panel_instance: LogPanel):
        self.log_panel_instance = log_panel_instance

    def write(self, message):
        self.log_panel_instance.write(message)

    def flush(self):
        """Required for stream interface."""
        pass

class Audio2MDUI:
    """
    Singleton UI manager providing consistent interface components

    Features:
    - Standardized progress bars
    - Unified logging configuration (including Live display)
    - Consistent error handling
    - Themed console output
    - Live display setup helper
    """

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._init_ui()
        return cls._instance

    def _init_ui(self):
        """Initialize UI components with consistent theme"""
        self.theme = Theme({
            "success": "green4",
            "warning": "gold3",
            "error": "red3",
            "info": "blue",
            "progress": "cyan",
            "metric": "magenta"
        })
        self.console = Console(theme=self.theme)
        self._setup_progress_columns()

    def _setup_progress_columns(self):
        """Configure standardized progress bar layout"""
        self.progress_columns = [
            TextColumn("[progress]{task.description}", justify="left"),
            BarColumn(bar_width=60),  # Set a fixed width for the BarColumn
            TextColumn("•"),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[phase]}"),
        ]

    def create_progress(self) -> Progress:
        """
        Create a pre-configured Progress instance

        Returns:
            Progress: Rich Progress bar with standard columns
        """
        # Ensure transient=False so progress bars persist after completion if needed
        return Progress(*self.progress_columns, console=self.console, transient=False)

    def configure_basic_logging(self) -> RichHandler:
        """
        Create standardized basic Rich logging handler (for non-Live scenarios).

        Returns:
            RichHandler: Pre-configured logging handler writing to the main console.
        """
        return RichHandler(
            show_time=True, # Show time for basic logging
            show_level=True,
            show_path=True, # Show path for basic logging
            console=self.console,
            markup=True,
            rich_tracebacks=True # Enable rich tracebacks
        )

    def setup_live_logging(self, log_panel_instance: LogPanel, debug_mode: bool, log_file_base: str):
        """
        Configure logging for Live display integration.

        Sets up file logging (if debug_mode) and a RichHandler directed
        to the provided LogPanel instance.

        Args:
            log_panel_instance: The LogPanel instance to capture logs.
            debug_mode: Boolean indicating if DEBUG level file logging is enabled.
            log_file_base: Base name for the log file (e.g., "audio2md").
        """
        log_level = logging.DEBUG if debug_mode else logging.INFO
        timestamp = datetime.datetime.now().strftime("%H-%M-%S")
        log_file = f"{log_file_base}_{timestamp}.log" if debug_mode else None
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level) # Set root logger level

        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                continue # Keep file handlers
            if isinstance(handler, RichHandler) and isinstance(handler.console.file, LogPanelStream):
                 if handler.console.file.log_panel_instance == log_panel_instance:
                     continue # Keep the correct RichHandler if already added
            root_logger.removeHandler(handler)
            handler.close() # Ensure handler resources are released

        # File Handler (Only add if DEBUG)
        if debug_mode:
            # Check if a file handler for this log file already exists
            file_handler_exists = any(
                isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file)
                for h in root_logger.handlers
            )
            if not file_handler_exists:
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
                try:
                    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
                    file_handler.setLevel(logging.DEBUG) # Log everything to file in debug
                    file_handler.setFormatter(formatter)
                    root_logger.addHandler(file_handler)
                    logging.info(f"DEBUG mode enabled. Logging detailed output to {log_file}")
                except Exception as e:
                    self.console.print(f"[error]Could not configure file logging to {log_file}: {e}[/]", style="error")
                    logging.error(f"File logging setup failed: {e}", exc_info=True)

        # Rich Console Handler (for LogPanel)
        # Check if the specific RichHandler for this LogPanel already exists
        rich_handler_exists = any(
            isinstance(h, RichHandler) and isinstance(h.console.file, LogPanelStream) and h.console.file.log_panel_instance == log_panel_instance
            for h in root_logger.handlers
        )

        if not rich_handler_exists:
            # show_warnings is NOT a valid argument for RichHandler
            rich_log_handler = RichHandler(
                show_time=False, # Time is less relevant in the live panel
                show_level=True,
                show_path=False, # Path is less relevant in the live panel
                markup=True,
                rich_tracebacks=True, # Use rich tracebacks
                log_time_format="[%X]" # Simple time format if shown
            )
            # Custom formatter to output only the message with markup for the panel
            log_formatter = logging.Formatter("%(message)s")
            rich_log_handler.setFormatter(log_formatter)

            # Use LogPanelStream to direct output
            # Create a dedicated console for this handler to avoid conflicts
            panel_console = Console(file=LogPanelStream(log_panel_instance), force_terminal=True, color_system="truecolor", theme=self.theme)
            rich_log_handler.console = panel_console
            # Set level based on debug mode: DEBUG for debug, INFO otherwise
            rich_log_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
            root_logger.addHandler(rich_log_handler)
            logging.info("Rich logging handler configured for Live display.")
        else:
             logging.debug("RichHandler for LogPanel already configured.")


    def create_live_display(self, log_panel_title: str = "[bold blue]Processing Status[/]", progress_panel_title: str = "[bold green]Progress[/]") -> Tuple[Live, Progress, LogPanel]:
        """
        Creates a standard Live display with Progress and LogPanel.

        Returns:
            Tuple[Live, Progress, LogPanel]: A tuple containing the configured
                                             Live object, Progress instance,
                                             and LogPanel instance.
        """
        log_panel = LogPanel(self.console) # Instantiate the log panel with console
        progress = self.create_progress() # Create the progress bar

        # Create the layout
        layout = Layout()
        layout.split_column(
            # Wrap progress in a Panel for title and border, centered, with minimum width
            Layout(Panel(progress, title=progress_panel_title, border_style="green", expand=True), name="progress"), # Added expand=True
            # Wrap log_panel instance in its own Panel for title/border
            Layout(log_panel, name="logs") # Removed Panel wrapper to eliminate border and title
        )

        # Create the Live object with improved overflow handling
        live = Live(layout, console=self.console, refresh_per_second=6, screen=False) # Removed vertical_overflow="ellipsis"

        return live, progress, log_panel

    def error_panel(self, message: str, exception: Optional[Exception] = None):
        """
        Display consistent error messaging using Rich Panel.

        Args:
            message: Primary error message.
            exception: Optional exception for traceback.
        """
        error_content = Text(message, style="bold red")
        if exception:
            # Use Rich's traceback formatting
            tb_text = self.console.extract_traceback(
                 (type(exception), exception, exception.__traceback__)
            )
            error_content.append("\n\nTraceback:\n")
            error_content.append(tb_text)

        self.console.print(Panel(error_content, title="[bold red]Error[/]", border_style="red", expand=False))


    def summary_table(self, data: Dict[str, str], title: str = "Processing Summary") -> Table:
        """
        Generate standardized summary table

        Args:
            data: Dictionary of metric names to values
            title: Title for the table

        Returns:
            Table: Formatted Rich table
        """
        table = Table(
            title=title,
            show_header=True,
            header_style="bold magenta"
        )
        table.add_column("Metric", style="metric", overflow="fold") # Allow folding long metric names
        table.add_column("Value", style="success", overflow="fold") # Allow folding long values

        for key, value in data.items():
            table.add_row(key, str(value))

        return table

    def get_console(self) -> Console:
        """
        Get the shared console instance

        Returns:
            Console: Themed console instance
        """
        return self.console

    def create_text(self) -> Text:
        """
        Create a new Text instance

        Returns:
            Text: Empty Text instance for building rich text
        """
        return Text()

    def create_panel(self, content, title: str, border_style: str = "blue", **kwargs) -> Panel:
        """
        Create a new Panel instance using the shared console theme.

        Args:
            content: Renderable content for the panel (e.g., Text, Table).
            title: Title for the panel (markup supported).
            border_style: Style for the panel border (default: "blue").
            **kwargs: Additional arguments for Panel (e.g., expand=False).

        Returns:
            Panel: Formatted Rich panel.
        """
        # Ensure title uses markup
        formatted_title = f"[{border_style}]{title}[/]"
        return Panel(content, title=formatted_title, border_style=border_style, **kwargs)
