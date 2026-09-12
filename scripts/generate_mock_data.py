"""Generate a comprehensive, realistic TREC-IS style crisis dataset for testing."""

import gzip
import json
from pathlib import Path
import random

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Comprehensive disaster events with varied incident templates
EVENTS_DATA = [
    {
        "event_id": "EVENT_FLOOD_RIVERVALE",
        "category": "SearchAndRescue",
        "priority": "Critical",
        "filename": "event_rivervale_flood.json.gz",
        "templates": [
            "CRITICAL: Flash flood water rising rapidly on River Road! 5 people including children trapped on roof of vehicle at mile 14! #RivervaleFlood #RescueNeeded",
            "Emergency: Family of 4 stranded inside house near River Road crossing. Water is at second floor window level! Please send rescue boats!",
            "Flood water sweeping through Elm Street neighborhood. Two elderly residents clinging to tree near bridge! Water rescue units needed urgently!",
            "RT @RivervalePD: Urgent evacuation notice for River Road basin. Search and rescue zodiac boats deployed to Sector 4. Do NOT attempt to drive through flood water!",
            "Water levels crossing 8 feet near River Road church. 3 people trapped in minivan near bridge! https://t.co/floodrescue99 #FloodSOS",
            "Update: First responders on scene with swift water rescue team at River Road bridge. Evacuations in progress for 12 stranded residents.",
            "Water rising over 2 inches per minute on River Road. 6 citizens unable to exit apartment ground floor! Urgent rescue required!",
            "Coast guard helicopter spotted over River Road basin winching trapped flood victims from rooftop. Strong currents reported."
        ]
    },
    {
        "event_id": "EVENT_EARTHQUAKE_CENTRAL",
        "category": "MedicalNeeds",
        "priority": "Critical",
        "filename": "event_central_earthquake.json.gz",
        "templates": [
            "URGENT: St. Jude Hospital emergency wing overwhelmed after M6.4 earthquake! Critically low on O-negative blood units and trauma surgery kits!",
            "Paramedics urgently requesting portable oxygen concentrators and burn dressing kits at Central Community Clinic. Multiple crush injuries incoming.",
            "Field hospital established at South High School football stadium. Immediate need for volunteer triage physicians, pediatric medicine, and sterile bandages.",
            "Mass casualty triage center active at City Pavilion. Treating 45 patients with fractures and lacerations. Sterile saline and IV kits urgently needed!",
            "City Hospital pharmacy collapsed. Insulin, asthma inhalers, and cardiac medications needed immediately for sheltered chronic patients. #DisasterMed",
            "Blood bank appeal: Donors needed tomorrow 7 AM at Blood Care Mobile units. Current hospital reserve is under 2 hours for trauma cases.",
            "Emergency dialysis equipment urgently requested at North Pavilion. Power generator fuel low. 14 critical kidney patients awaiting treatment.",
            "Medical dispatch: Ambulances unable to reach downtown due to rubble. Need volunteer EMTs with portable trauma backpacks on foot."
        ]
    },
    {
        "event_id": "EVENT_WILDFIRE_PINEVALLEY",
        "category": "InfrastructureDamage",
        "priority": "High",
        "filename": "event_pinevalley_fire.json.gz",
        "templates": [
            "Major structural damage: Pine Valley Bridge on Highway 101 cracked and buckling from fire heat. Highway closed in both directions until further notice.",
            "Wildfire knocked down 4 main transmission towers. Over 35,000 households without electricity in Western Valley. Power grid repair crews blocked by fire.",
            "Water treatment plant near Pine Valley reservoir sustained pump electrical failure. Tap water contamination alert issued! Boil water notice in effect.",
            "Cellular towers down across all of Westview county. Emergency responders relying exclusively on HAM radio relay frequencies 146.520 MHz.",
            "Natural gas main rupture reported near 8th Street industrial park after falling burning pine trees. Fire crews requesting emergency gas valve shutdown.",
            "Railway tracks severely warped by intense ground heat near mile marker 82. Freight line traffic suspended between North and South terminals.",
            "Severe smoke and asphalt melting reported on Ridge Road. Road surface impassable for light emergency vehicles. Heavy bulldozers en route.",
            "Structural safety inspection: North Elementary School building is NOT collapsed, but structural roof inspection is ongoing. School remains closed."
        ]
    },
    {
        "event_id": "EVENT_HURRICANE_COASTAL",
        "category": "AffectedPopulation",
        "priority": "Medium",
        "filename": "event_coastal_hurricane.json.gz",
        "templates": [
            "Over 1,200 coastal evacuees currently registered at Memorial Convention Center shelter. Hot meals, cots, and dry hygiene kits being distributed.",
            "Displaced families arriving at Westside Church gymnasium shelter. Cots are full; second intake shelter opening at East High School tonight.",
            "Missing person alert: 74-year-old grandmother last seen wearing blue raincoat during evacuation from Bayfront Avenue. Contact dispatch with info.",
            "Pet-friendly temporary shelter opened at Fairgrounds Pavilion 3 for residents evacuating with dogs and cats. Food and water bowls available.",
            "Red Cross family reunification desk set up at Central Arena. Please check in online or visit desk if looking for separated relatives. #FamilyFind",
            "Community liaison: 85 foreign tourists stranded at Harbor Hotel without transport. Shuttle buses dispatched to transport to airport evacuation hub.",
            "Evacuee welfare update: Clean water dispensers and phone charging stations installed at South Shelter entrance. Safe zones established."
        ]
    },
    {
        "event_id": "EVENT_RELIEF_HARBOR",
        "category": "DonationsAndVolunteering",
        "priority": "Low",
        "filename": "event_harbor_relief.json.gz",
        "templates": [
            "Relief donation warehouse open at Gate 4, Port Logistics Park. Accepting canned food, bottled water, baby diapers, and unopened hygiene supplies.",
            "Volunteer callout: 50 volunteers needed tomorrow morning at 8:30 AM to assist with sorting and palletizing emergency relief packages.",
            "Massive thank you to local bakeries and restaurants providing 500 hot meals to emergency utility crews and rescue workers today! #CommunityHeroes",
            "Pet rescue network seeking foster homes for 30 displaced shelter dogs and cats while families transition to temporary housing.",
            "Donation center announcement: We currently have enough clothing. High demand now is baby formula, portable power banks, and heavy-duty work gloves.",
            "Weekend volunteer cleanup brigade organizing Sunday 9 AM at Riverfront Park. Bring rubber boots, shovels, and work gloves. #CleanupSquad",
            "Monetary donation link established via verified disaster foundation. 100% of proceeds go to emergency food hampers and shelter vouchers."
        ]
    }
]


def generate_comprehensive_dataset(tweets_per_event: int = 25):
    """Generate multiple realistic gzipped tweet files and comprehensive labels.json."""
    all_labels = []
    total_tweets_count = 0
    random.seed(42)

    # Clean existing raw files
    for old_file in RAW_DIR.glob("*.json.gz"):
        old_file.unlink()

    for event in EVENTS_DATA:
        event_tweets = []
        event_id = event["event_id"]
        category = event["category"]
        priority = event["priority"]
        templates = event["templates"]
        filename = event["filename"]

        for i in range(tweets_per_event):
            tweet_idx = total_tweets_count + i + 1
            tweet_id = f"tweet_{1000 + tweet_idx}"
            base_template = templates[i % len(templates)]

            # Introduce realistic variations: handles, retweets, timestamps
            user_id = f"user_{random.randint(100, 999)}"
            hour = (10 + (i // 3)) % 24
            minute = random.randint(10, 59)
            created_at = f"Sat Sep 12 {hour:02d}:{minute:02d}:30 +0000 2026"

            if i % 4 == 0:
                tweet_text = f"RT @emergency_intel: {base_template}"
            elif i % 5 == 0:
                tweet_text = f"{base_template} https://t.co/disaster{tweet_idx:04d}"
            else:
                tweet_text = f"{base_template}"

            tweet_record = {
                "id_str": tweet_id,
                "id": int(tweet_id.replace("tweet_", "")),
                "full_text": tweet_text,
                "text": tweet_text,
                "user": {
                    "id_str": user_id,
                    "screen_name": f"first_responder_{user_id}",
                    "verified": (i % 3 == 0)
                },
                "created_at": created_at,
                "lang": "en"
            }
            event_tweets.append(tweet_record)

            all_labels.append({
                "tweet_id": tweet_id,
                "categories": [category],
                "priority": priority,
                "cluster_id": event_id
            })

        total_tweets_count += tweets_per_event

        # Save event-specific gzipped JSON
        file_path = RAW_DIR / filename
        with gzip.open(file_path, "wt", encoding="utf-8") as f:
            for t in event_tweets:
                f.write(json.dumps(t) + "\n")
        print(f"Generated {len(event_tweets)} tweets in '{file_path}' (Event: {event_id})")

    # Save comprehensive labels.json
    labels_path = RAW_DIR / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent=2)

    print(f"\nSuccessfully generated {total_tweets_count} crisis reports across {len(EVENTS_DATA)} events.")
    print(f"Labels cataloged at '{labels_path}'.")


if __name__ == "__main__":
    generate_comprehensive_dataset(tweets_per_event=25)
