import httpx
from typing import Optional
import logging

logger = logging.getLogger(__name__)

COUNTRIES_API_URL = "https://restcountries.com/v3.1/name/{country}"


async def fetch_country_data(country_name: str) -> Optional[dict]:
    """
    Fetches country data from the REST Countries public API.

    Returns the first matching country as a dict, or None if:
    - Country not found (404)
    - Network/timeout error
    - Unexpected response shape
    """
    url = COUNTRIES_API_URL.format(country=country_name.strip())

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url)

            if response.status_code == 404:
                logger.warning(f"Country not found in API: '{country_name}'")
                return None

            response.raise_for_status()
            data = response.json()

            if not isinstance(data, list) or len(data) == 0:
                logger.warning(f"Empty result for country: '{country_name}'")
                return None

            return data[0]

        except httpx.TimeoutException:
            logger.error(f"Timeout fetching country data for: '{country_name}'")
            raise RuntimeError("The countries API timed out. Please try again.")

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error for '{country_name}': {e}")
            raise RuntimeError(f"Countries API returned an error: {e.response.status_code}")

        except httpx.HTTPError as e:
            logger.error(f"Network error for '{country_name}': {e}")
            raise RuntimeError("Network error reaching the countries API.")