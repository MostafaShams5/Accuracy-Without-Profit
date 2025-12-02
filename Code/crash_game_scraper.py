# Description:
#   This script automates the collection of live game data from the 1xBet "Crash" game.
#   It uses the Playwright library to launch a browser, navigate to the game's URL,
#   and monitor the game's multiplier element.
#
#   The core logic identifies when a game round concludes (a "crash") and logs the
#   final multiplier, a timestamp, and a unique round ID to a CSV file. The script
#   is designed for continuous, long-term data acquisition.
#
# Output:
#   - dataset_crash_game_raw.csv: A CSV file containing the raw, timestamped crash data.
#
import asyncio
import csv
from datetime import datetime
from playwright.async_api import async_playwright, TimeoutError

# --- Configuration ---
URL = "https://1xbet.com/en/games/crash"
IFRAME_SELECTOR = "iframe.games-project-frame__item"
COUNTER_SELECTOR = "text.crash-game__counter"
OUTPUT_FILE = "1xbet_crash_data.csv"

# --- Helper Functions ---
def setup_csv():
    # gotta check if the CSV exists, or if we're starting fresh.
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            last_row = None
            for row in reader:
                last_row = row
            if header is None:
                raise FileNotFoundError
        if last_row:
            return int(last_row[0]) # pick up where we left off.
    except (FileNotFoundError, IndexError):
        # file is new or empty, so we create it.
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['roundID', 'timestamp', 'crash'])
    return 0

async def run_scraper():
    """The final, robust data logger for the Crash game."""
    async with async_playwright() as p:
        print("Launching browser with Playwright...")
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = await context.new_page()
        print("Browser launched.")

        # pull the last round ID from the file.
        round_id = setup_csv()

        with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)

            try:
                print(f"Navigating to: {URL}")
                await page.goto(URL, timeout=90000)
                print("Page loaded.")

                print("Waiting for the game iframe...")
                iframe_locator = page.frame_locator(IFRAME_SELECTOR)
                print("Successfully located game iframe.")

                print("Looking for the game counter element...")
                counter_locator = iframe_locator.locator(COUNTER_SELECTOR)
                await counter_locator.wait_for(state="visible", timeout=90000)
                print(f"Success! Found '{COUNTER_SELECTOR}'. Starting data logging...")

                # a simple little state machine to track the game.
                game_state = "WAITING"
                last_known_multiplier = "1.00x"

                while True:
                    current_text = (await counter_locator.text_content()).strip()

                    # rocket is flying, update the last known multiplier.
                    if 'x' in current_text and len(current_text) > 1:
                        if game_state == "WAITING":
                            game_state = "FLYING"
                        last_known_multiplier = current_text

                    # crash! time to log it.
                    elif current_text == 'x' and game_state == "FLYING":
                        round_id += 1
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        clean_multiplier = last_known_multiplier.replace('x', '')

                        csv_writer.writerow([round_id, timestamp, clean_multiplier])
                        f.flush() # write to disk immediately.

                        print(f"Logged Round {round_id}: {clean_multiplier} at {timestamp}")

                        game_state = "WAITING"

                    await page.wait_for_timeout(100)

            except TimeoutError:
                print("\nFATAL ERROR: A required element was not found after 90 seconds.")
            except KeyboardInterrupt:
                print("\nLogging stopped by user.")
            finally:
                await browser.close()
                print(f"Browser closed. Data saved to {OUTPUT_FILE}")

async def main():
    await run_scraper()

if __name__ == "__main__":
    asyncio.run(main())
