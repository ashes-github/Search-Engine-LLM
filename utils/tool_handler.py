import logging

logger = logging.getLogger(__name__)


def safe_tool_call(tool_name: str, func, query: str) -> str:

    try:
        logger.info(f"{tool_name} called: {query}")

        result = func(query)

        logger.info(f"{tool_name} call completed successfully")

        return result

    except Exception:

        logger.exception(f"{tool_name} call failed after retries: {query}")

        return (
            f"{tool_name} search failed temporarily. "
            "Try another source or answer from general knowledge."
        )
