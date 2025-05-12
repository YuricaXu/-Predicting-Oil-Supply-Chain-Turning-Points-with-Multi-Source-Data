import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import json
import os

def create_opec_events_data():
    """
    Create a dataset of OPEC events (manually organized data)
    """
    opec_events = [
        # Format: (date, event_type, description)
        ('2008-09-10', 'meeting', 'OPEC meeting - maintain production'),
        ('2008-10-24', 'cut', 'OPEC announces 1.5M b/d cut'),
        ('2008-12-17', 'cut', 'OPEC announces 2.2M b/d cut'),
        ('2009-03-15', 'maintain', 'OPEC maintains production'),
        ('2009-05-28', 'maintain', 'OPEC maintains production'),
        ('2009-09-09', 'maintain', 'OPEC maintains production'),
        ('2009-12-22', 'maintain', 'OPEC maintains production'),
        ('2010-03-17', 'maintain', 'OPEC maintains production'),
        ('2010-10-14', 'maintain', 'OPEC maintains production'),
        ('2010-12-11', 'maintain', 'OPEC maintains production'),
        ('2011-06-08', 'no_agreement', 'OPEC fails to reach agreement'),
        ('2011-12-14', 'maintain', 'OPEC maintains production'),
        ('2012-06-14', 'maintain', 'OPEC maintains production'),
        ('2012-12-12', 'maintain', 'OPEC maintains production'),
        ('2013-05-31', 'maintain', 'OPEC maintains production'),
        ('2013-12-04', 'maintain', 'OPEC maintains production'),
        ('2014-06-11', 'maintain', 'OPEC maintains production'),
        ('2014-11-27', 'maintain', 'OPEC maintains production'),
        ('2015-06-05', 'maintain', 'OPEC maintains production'),
        ('2015-12-04', 'maintain', 'OPEC maintains production'),
        ('2016-06-02', 'maintain', 'OPEC maintains production'),
        ('2016-11-30', 'cut', 'OPEC agrees to cut production by 1.2M b/d'),
        ('2017-05-25', 'extend', 'OPEC extends cuts for 9 months'),
        ('2017-11-30', 'extend', 'OPEC extends cuts until end of 2018'),
        ('2018-06-22', 'increase', 'OPEC agrees to increase production'),
        ('2018-12-07', 'cut', 'OPEC agrees to cut production by 1.2M b/d'),
        ('2019-07-01', 'extend', 'OPEC extends cuts for 9 months'),
        ('2019-12-06', 'cut', 'OPEC agrees to cut production by 500K b/d'),
        ('2020-03-06', 'no_agreement', 'OPEC+ fails to reach agreement'),
        ('2020-04-12', 'cut', 'OPEC+ agrees to cut production by 9.7M b/d'),
        ('2020-06-06', 'extend', 'OPEC+ extends cuts for 1 month'),
        ('2020-12-03', 'increase', 'OPEC+ agrees to increase production'),
        ('2021-01-04', 'maintain', 'OPEC+ maintains production'),
        ('2021-03-04', 'maintain', 'OPEC+ maintains production'),
        ('2021-04-01', 'increase', 'OPEC+ agrees to increase production'),
        ('2021-07-01', 'increase', 'OPEC+ agrees to increase production'),
        ('2021-09-01', 'maintain', 'OPEC+ maintains production'),
        ('2021-10-04', 'increase', 'OPEC+ agrees to increase production'),
        ('2021-12-02', 'maintain', 'OPEC+ maintains production'),
        ('2022-01-04', 'increase', 'OPEC+ agrees to increase production'),
        ('2022-02-02', 'maintain', 'OPEC+ maintains production'),
        ('2022-03-02', 'increase', 'OPEC+ agrees to increase production'),
        ('2022-04-01', 'maintain', 'OPEC+ maintains production'),
        ('2022-05-05', 'increase', 'OPEC+ agrees to increase production'),
        ('2022-06-02', 'increase', 'OPEC+ agrees to increase production'),
        ('2022-07-01', 'maintain', 'OPEC+ maintains production'),
        ('2022-08-03', 'increase', 'OPEC+ agrees to increase production'),
        ('2022-09-05', 'cut', 'OPEC+ agrees to cut production by 100K b/d'),
        ('2022-10-05', 'cut', 'OPEC+ agrees to cut production by 2M b/d'),
        ('2022-11-01', 'maintain', 'OPEC+ maintains production'),
        ('2022-12-04', 'maintain', 'OPEC+ maintains production'),
        ('2023-01-04', 'maintain', 'OPEC+ maintains production'),
        ('2023-02-01', 'maintain', 'OPEC+ maintains production'),
        ('2023-03-01', 'maintain', 'OPEC+ maintains production'),
        ('2023-04-03', 'cut', 'OPEC+ announces surprise cut of 1.16M b/d'),
        ('2023-06-04', 'cut', 'OPEC+ agrees to extend cuts'),
        ('2023-07-01', 'maintain', 'OPEC+ maintains production'),
        ('2023-08-01', 'maintain', 'OPEC+ maintains production'),
        ('2023-09-05', 'extend', 'OPEC+ extends cuts until end of 2023'),
        ('2023-10-04', 'maintain', 'OPEC+ maintains production'),
        ('2023-11-30', 'cut', 'OPEC+ agrees to cut production by 2.2M b/d'),
        ('2023-12-01', 'maintain', 'OPEC+ maintains production'),
        ('2024-01-01', 'maintain', 'OPEC+ maintains production'),
        ('2024-02-01', 'maintain', 'OPEC+ maintains production'),
        ('2024-03-03', 'extend', 'OPEC+ extends cuts until end of Q2 2024'),
    ]
    # Convert to DataFrame
    df_events = pd.DataFrame(opec_events, columns=['date', 'event_type', 'description'])
    df_events['date'] = pd.to_datetime(df_events['date'])
    return df_events

def scrape_opec_events():
    """
    Scrape OPEC event data from MacroMicro website. If scraping fails, fallback to manual data.
    """
    url = "https://en.macromicro.me/time_line?id=21&stat=389"
    try:
        # Prepare request headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Referer': 'https://en.macromicro.me/',
        }
        print("Sending request to MacroMicro...")
        session = requests.Session()
        response = session.get(url, headers=headers)
        response.raise_for_status()
        print("Parsing HTML response...")
        soup = BeautifulSoup(response.text, 'html.parser')
        # Print page title for debugging
        print("Page title:", soup.title.text if soup.title else "No title found")
        # Save raw HTML for debugging
        os.makedirs('data/debug', exist_ok=True)
        with open('data/debug/opec_page.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        events = []
        # Try to find timeline elements (may not work if page is dynamic)
        timeline = soup.find('div', class_='timeline')
        if timeline:
            print("Found timeline element")
            event_elements = timeline.find_all('div', class_='event')
            print(f"Found {len(event_elements)} events")
            for event in event_elements:
                try:
                    date_elem = event.find('div', class_='date')
                    desc_elem = event.find('div', class_='description')
                    if date_elem and desc_elem:
                        date_str = date_elem.text.strip()
                        description = desc_elem.text.strip()
                        # Infer event type from description
                        event_type = 'meeting'
                        if 'cut' in description.lower():
                            event_type = 'cut'
                        elif 'increase' in description.lower():
                            event_type = 'increase'
                        elif 'maintain' in description.lower():
                            event_type = 'maintain'
                        elif 'extend' in description.lower():
                            event_type = 'extend'
                        elif 'no agreement' in description.lower():
                            event_type = 'no_agreement'
                        try:
                            date = datetime.strptime(date_str, '%Y-%m-%d')
                            events.append((date, event_type, description))
                            print(f"Added event: {date_str} - {description}")
                        except ValueError:
                            print(f"Could not parse date: {date_str}")
                except Exception as e:
                    print(f"Error processing event: {str(e)}")
        # If no events found, try to find JSON data in script tags
        if not events:
            print("Trying to find JSON data in script tags...")
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'timelineData' in script.string:
                    print("Found script tag that may contain data")
                    # Further parsing could be added here
        # Create DataFrame if events were found
        if events:
            df_events = pd.DataFrame(events, columns=['date', 'event_type', 'description'])
            df_events = df_events.sort_values('date')
            os.makedirs('data', exist_ok=True)
            df_events.to_csv('data/opec_events.csv', index=False)
            print(f"Successfully scraped {len(events)} OPEC events.")
            return df_events
        else:
            print("No event data found on the page.")
            return None
    except Exception as e:
        print(f"Error occurred during scraping: {str(e)}")
        return None

def main():
    print("Starting OPEC event data scraping...")
    df_events = scrape_opec_events()
    if df_events is not None:
        print("\nPreview of OPEC event data:")
        print(df_events.head())
        print("\nEvent type counts:")
        print(df_events['event_type'].value_counts())
    else:
        print("Scraping failed, using manually organized data.")
        df_events = create_opec_events_data()
        print("\nPreview of manually organized OPEC event data:")
        print(df_events.head())
        print("\nEvent type counts:")
        print(df_events['event_type'].value_counts())
        # Save the manual data for downstream use
        os.makedirs('data', exist_ok=True)
        df_events.to_csv('data/opec_events.csv', index=False)
        print("\nManually organized data saved to data/opec_events.csv")

if __name__ == "__main__":
    main() 