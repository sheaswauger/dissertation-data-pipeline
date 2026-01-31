#!/usr/bin/env python3
"""
Text Processing Utilities
Handles HTML→TXT conversion and keyword searching.
"""

import base64
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from bs4 import BeautifulSoup
import PyPDF2
import io


class TextProcessor:
    """Process bill texts for extraction and searching"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def decode_bill_text(self, doc_base64: str, mime_type: str) -> Tuple[bytes, str]:
        """
        Decode base64 bill text and determine file extension
        Returns: (decoded_bytes, file_extension)
        """
        if not doc_base64:
            raise ValueError("Empty document base64")

        doc_bytes = base64.b64decode(doc_base64)

        ext_map = {
            'text/html': '.html',
            'application/pdf': '.pdf',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'text/plain': '.txt'
        }
        ext = ext_map.get(mime_type, '.txt')

        return doc_bytes, ext

    def html_to_text(self, html_bytes: bytes) -> str:
        """Convert HTML to plain text"""
        try:
            html_content = html_bytes.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html_content, 'lxml')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text(separator='\n', strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines()]
            text = '\n'.join(line for line in lines if line)

            return text

        except Exception as e:
            self.logger.warning(f"HTML parsing failed, falling back to raw decode: {e}")
            return html_bytes.decode('utf-8', errors='ignore')

    def pdf_to_text(self, pdf_bytes: bytes) -> str:
        """Convert PDF to plain text"""
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text_parts = []
            for page in pdf_reader.pages:
                text_parts.append(page.extract_text())

            return '\n'.join(text_parts)

        except Exception as e:
            self.logger.warning(f"PDF parsing failed: {e}")
            return ""

    def extract_text_from_bytes(self, doc_bytes: bytes, file_ext: str) -> str:
        """
        Extract plain text from document bytes based on file extension
        Returns plain text string
        """
        if file_ext == '.html':
            return self.html_to_text(doc_bytes)
        elif file_ext == '.pdf':
            return self.pdf_to_text(doc_bytes)
        elif file_ext in ['.txt', '.doc', '.docx']:
            # For .txt, just decode
            # For .doc/.docx, we'd need additional libraries (python-docx)
            # For now, just decode as text
            return doc_bytes.decode('utf-8', errors='ignore')
        else:
            return doc_bytes.decode('utf-8', errors='ignore')

    def search_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """
        Search for keywords in text (case-insensitive, substring match)
        Returns list of matched keywords
        """
        text_lower = text.lower()
        matched = []

        for keyword in keywords:
            if keyword.lower() in text_lower:
                matched.append(keyword)

        return matched

    def save_bill_text(
        self,
        doc_bytes: bytes,
        file_ext: str,
        state: str,
        bill_number: str,
        html_dir: Path,
        txt_dir: Path
    ) -> Tuple[str, str]:
        """
        Save bill text in both HTML/original format and TXT format
        Returns: (html_filename, txt_filename)
        """
        # Create safe filename
        safe_bill = bill_number.replace('/', '_').replace('\\', '_').replace(' ', '_')
        base_filename = f"{state}_{safe_bill}"

        # Save original format (HTML, PDF, etc.)
        html_filename = base_filename + file_ext
        html_path = html_dir / html_filename

        try:
            with open(html_path, 'wb') as f:
                f.write(doc_bytes)
            self.logger.debug(f"Saved original: {html_filename}")
        except Exception as e:
            self.logger.error(f"Failed to save original file {html_filename}: {e}")
            html_filename = ""

        # Extract and save plain text
        txt_filename = base_filename + '.txt'
        txt_path = txt_dir / txt_filename

        try:
            text = self.extract_text_from_bytes(doc_bytes, file_ext)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            self.logger.debug(f"Saved text: {txt_filename}")
        except Exception as e:
            self.logger.error(f"Failed to save text file {txt_filename}: {e}")
            txt_filename = ""

        return html_filename, txt_filename
