"""Notion reader. Taken and modified from https://github.com/jerryjliu/gpt_index/blob/main/gpt_index/readers/notion.py"""
import os
from typing import Any, Dict, List, Optional

import requests

from gpt_index.readers.base import BaseReader
from gpt_index.schema import Document

INTEGRATION_TOKEN_NAME = "NOTION_INTEGRATION_TOKEN"
BLOCK_CHILD_URL_TMPL = "https://api.notion.com/v1/blocks/{block_id}/children"
SEARCH_URL = "https://api.notion.com/v1/search"


# TODO: Notion DB reader coming soon!
class NotionPageReader(BaseReader):
    """Notion Page reader.
    Reads a set of Notion pages.
    """

    def __init__(self, integration_token: Optional[str] = None) -> None:
        """Initialize with parameters."""
        if integration_token is None:
            integration_token = os.getenv(INTEGRATION_TOKEN_NAME)
            if integration_token is None:
                raise ValueError(
                    "Must specify `integration_token` or set environment "
                    "variable `NOTION_INTEGRATION_TOKEN`."
                )
        self.token = integration_token
        self.headers = {
            "Authorization": "Bearer " + self.token,
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

    def _read_block(self, block_id: str, num_tabs: int = 0) -> str:
        """Read a block."""
        done = False
        result_lines_arr = []
        cur_block_id = block_id
        while not done:
            block_url = BLOCK_CHILD_URL_TMPL.format(block_id=cur_block_id)
            query_dict: Dict[str, Any] = {}

            res = requests.request(
                "GET", block_url, headers=self.headers, json=query_dict
            )
            data = res.json()

            for result in data["results"]:
                result_type = result["type"]
                result_obj = result[result_type]
                # NOTE: Notion reader doesn't support all block objects atm, only
                # block objects with rich text.
                if "rich_text" not in result_obj:
                    continue

                cur_result_text_arr = []
                for rich_text in result_obj["rich_text"]:
                    # skip if doesn't have text object
                    if "text" in rich_text:
                        text = rich_text["text"]["content"]
                        prefix = "\t" * num_tabs
                        cur_result_text_arr.append(prefix + text)

                result_block_id = result["id"]
                has_children = result["has_children"]
                if has_children:
                    children_text = self._read_block(
                        result_block_id, num_tabs=num_tabs + 1
                    )
                    cur_result_text_arr.append(children_text)

                cur_result_text = "\n".join(cur_result_text_arr)
                result_lines_arr.append(cur_result_text)

            if data["next_cursor"] is None:
                done = True
                break
            else:
                cur_block_id = data["next_cursor"]

        result_lines = "\n".join(result_lines_arr)
        return result_lines

    def read_page(self, page_id: str) -> str:
        """Read a page."""
        return self._read_block(page_id)

    def search(self, query: str) -> List[str]:
        """Search Notion page given a text query."""
        done = False
        next_cursor: Optional[str] = None
        results = []
        while not done:
            query_dict = {
                "query": query,
            }
            if next_cursor is not None:
                query_dict["start_cursor"] = next_cursor
            res = requests.post(SEARCH_URL, headers=self.headers, json=query_dict)
            data = res.json()
            for result in data["results"]:
                result_data = {
                    "page_id": result["id"],
                    "created_by": result["created_by"]["id"],
                    "last_edited_time": result["last_edited_time"],
                    "source": result["url"]
                }
                results.append(result_data)

            if data["next_cursor"] is None:
                done = True
                break
            else:
                next_cursor = data["next_cursor"]
        return results

    def load_data(self, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""
        docs = []
        for metadata in load_kwargs["pages"]:
            page_text = self.read_page(metadata["page_id"])
            docs.append(Document(page_text, extra_info=metadata))
        return docs


if __name__ == "__main__":
    reader = NotionPageReader()
    print(reader.search("What I"))