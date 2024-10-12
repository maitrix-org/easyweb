parse_flight_description_prompt = """\
Parse out the dimensions and values of fight descriptions into JSON format following the examples below:

Example 1:

Description: From 13006 US dollars round trip total. Nonstop flight with United. Leaves Chicago O'Hare International Airport at 12:40\\u202fPM on Sunday, October 20 and arrives at Haneda Airport at 3:55\\u202fPM on Monday, October 21. Total duration 13 hr 15 min.

Response:
```
{
  "price": 13006,
  "currency": "USD",
  "flight_type": "nonstop",
  "airline": "United",
  "departure_city": "Chicago",
  "departure_airport": "O'Hare International Airport",
  "departure_time": "12:40 PM",
  "departure_day": "Sunday",
  "departure_date": "October 20",
  "arrival_city": "Tokyo",
  "arrival_airport": "Haneda Airport",
  "arrival_time": "3:55 PM",
  "arrival_day": "Monday",
  "arrival_date": "October 21",
  "total_duration": "13 hr 15 min"
}
```

Example 2:

Description: From 8332 US dollars round trip total. 1 stop flight with Air Canada. Leaves Chicago O'Hare International Airport at 8:30\\u202fAM on Sunday, October 20 and arrives at Narita International Airport at 3:25\\u202fPM on Monday, October 21. Total duration 16 hr 55 min. Layover (1 of 1) is a 1 hr 5 min layover at Montréal-Pierre Elliott Trudeau International Airport in Montreal.

Response:
```
{
  "price": 8332,
  "currency": "USD",
  "flight_type": "1 stop",
  "airline": "Air Canada",
  "departure_city": "Chicago",
  "departure_airport": "O'Hare International Airport",
  "departure_time": "8:30 AM",
  "departure_day": "Sunday",
  "departure_date": "October 20",
  "arrival_city": "Tokyo",
  "arrival_airport": "Narita International Airport",
  "arrival_time": "3:25 PM",
  "arrival_day": "Monday",
  "arrival_date": "October 21",
  "total_duration": "16 hr 55 min",
  "layovers": [
    {
      "layover_city": "Montreal",
      "layover_airport": "Montréal-Pierre Elliott Trudeau International Airport",
      "layover_duration": "1 hr 5 min",
      "layover_count": 1
    }
  ]
}
```
"""

parse_flight_message_prompt = """\
Parse out the dimensions and values of the flights mentioned in the following messages into JSON format following the examples below. \
Format your response in a Python list. Don't include information not outlined in the JSON schema.

Example 1:

Message: I found two nonstop business class flights from Chicago to Tokyo. \
The first flight is with United, departing at 12:40 PM on October 20th from Chicago O\'Hare International Airport (ORD) \
and arriving at 3:55 PM on October 21st at Haneda Airport (HND). The total travel time is 13 hours and 15 minutes, \
and the flight costs $13,006. The second flight is with ANA, departing at 11:50 AM on October 20th from ORD and arriving \
at 3:00 PM on October 21st at HND. The total travel time is 13 hours and 10 minutes, and the flight costs $13,007. \
Both flights offer lie-flat seats and Wi-Fi for a fee.

Response:
```python
flights = [
    {
        "price": 13006,
        "currency": "USD",
        "flight_type": "nonstop",
        "airline": "United",
        "departure_city": "Chicago",
        "departure_airport": "O'Hare International Airport",
        "departure_time": "12:40 PM",
        "departure_day": None,
        "departure_date": "October 20",
        "arrival_city": "Tokyo",
        "arrival_airport": "Haneda Airport",
        "arrival_time": "3:55 PM",
        "arrival_day": None,
        "arrival_date": "October 21",
        "total_duration": "13 hr 15 min"
    },
    {
        "price": 13007,
        "currency": "USD",
        "flight_type": "nonstop",
        "airline": "ANA",
        "departure_city": "Chicago",
        "departure_airport": "O'Hare International Airport",
        "departure_time": "11:50 AM",
        "departure_day": None,
        "departure_date": "October 20",
        "arrival_city": "Tokyo",
        "arrival_airport": "Haneda Airport",
        "arrival_time": "3:00 PM",
        "arrival_day": None,
        "arrival_date": "October 21",
        "total_duration": "13 hr 10 min"
    }
]
```

Example 2:

Message: I found two nonstop business class flights from Chicago to Tokyo. \
The first flight is with United, departing at 12:40 PM on October 20th from Chicago O\'Hare International Airport (ORD) \
and arriving at 3:55 PM on October 21st at Haneda Airport (HND). The total travel time is 13 hours and 15 minutes, \
and the flight costs $13,006. The second flight is with ANA, departing at 11:50 AM on October 20th from ORD and arriving at \
3:00 PM on October 21st at HND. The total travel time is 13 hours and 10 minutes, and the flight costs $13,007. \
Both flights offer lie-flat seats and Wi-Fi for a fee.

Response:
```python
flights = [
    {
        "price": 149,
        "currency": "USD",
        "flight_type": "nonstop",
        "airline": "Spirit",
        "departure_city": "Dallas",
        "departure_airport": None,
        "departure_time": "7:39 AM",
        "departure_day": None,
        "departure_date": "October 9",
        "arrival_city": "Miami",
        "arrival_airport": None,
        "arrival_time": "11:36 AM",
        "arrival_day": None,
        "arrival_date": "October 9",
        "total_duration": None
    },
    {
        "price": 149,
        "currency": "USD",
        "flight_type": "nonstop",
        "airline": "Spirit",
        "departure_city": "Dallas",
        "departure_airport": None,
        "departure_time": "2:30 PM",
        "departure_day": None,
        "departure_date": "October 9",
        "arrival_city": "Miami",
        "arrival_airport": None,
        "arrival_time": "6:24 PM",
        "arrival_day": None,
        "arrival_date": "October 9",
        "total_duration": None
    },
    {
        "price": 159,
        "currency": "USD",
        "flight_type": "nonstop",
        "airline": "Frontier",
        "departure_city": "Dallas",
        "departure_airport": None,
        "departure_time": "8:49 PM",
        "departure_day": None,
        "departure_date": "October 9",
        "arrival_city": "Miami",
        "arrival_airport": None,
        "arrival_time": "12:31 AM",
        "arrival_day": None,
        "arrival_date": "October 10",
        "total_duration": None
    }
]
```

Example 3:

Message: The round-trip flight with the shortest layover from Toronto to Athens, departing on April 15th and returning on April 30th, \
is with Air France. The flight departs Toronto Pearson International Airport (YYZ) at 11:00 PM on April 15 and \
arrives at Athens International Airport (ATH) at 6:00 PM on April 16, with a 1 hr 30 min layover in Paris Charles de Gaulle Airport (CDG). \
The first leg is on an Airbus A350 in Premium Economy class with flight number AF 387. The total cost is $1,350 round trip.

Response:
```python
flights = [
    {
        "price": 1350,
        "currency": "USD",
        "flight_type": "1 stop",
        "airline": "Air France",
        "departure_city": "Toronto",
        "departure_airport": "Toronto Pearson International Airport",
        "departure_time": "11:00 PM",
        "departure_day": None,
        "departure_date": "April 15",
        "arrival_city": "Athens",
        "arrival_airport": "Athens International Airport",
        "arrival_time": "6:00 PM",
        "arrival_day": None,
        "arrival_date": "April 16",
        "total_duration": None,
        "layovers": [
            {
                "layover_city": "Paris",
                "layover_airport": "Charles de Gaulle Airport",
                "layover_duration": "1 hr 30 min",
                "layover_count": 1
            }
        ]
    }
]
```
"""

verify_flight_constraint_prompt = """\
My friend did flight searches with some criteria and received lists of flight options. Among those flights, my friend choose one. \
Your task is to verify whether the flight satisfies the provided constraints. \
Your response will be parsed by a program, so make sure to follow the exact format in the examples below.

Example 1:

Date and Time: Mon, Oct 07, 2024 23:16:08

Search Criteria:
- Ticket Type: Round trip
- Seating Class: Premium economy
- Origin: Toronto
- Destination: Athens
- Departure: Tue, Apr 15
- Return: Wed, Apr 30
- Sorting Criterion: Best flights
- Num Passengers: 1 passenger

Top flights based on search criteria:
[
  {
    "price": 1350,
    "currency": "USD",
    "flight_type": "1 stop",
    "airline": "Air France",
    "departure_city": "Toronto",
    "departure_airport": "Pearson International Airport",
    "departure_time": "11:00 PM",
    "departure_day": "Tuesday",
    "departure_date": "April 15",
    "arrival_city": "Athens",
    "arrival_airport": "Athens International Airport \"Eleftherios Venizelos\"",
    "arrival_time": "6:00 PM",
    "arrival_day": "Wednesday",
    "arrival_date": "April 16",
    "total_duration": "12 hr",
    "layovers": [
      {
        "layover_city": "Paris",
        "layover_airport": "Paris Charles de Gaulle Airport",
        "layover_duration": "1 hr 30 min",
        "layover_count": 1
      }
    ]
  },
  {
    "price": 1562,
    "currency": "USD",
    "flight_type": "1 stop",
    "airline": "Air Canada",
    "departure_city": "Toronto",
    "departure_airport": "Pearson International Airport",
    "departure_time": "4:00 PM",
    "departure_day": "Tuesday",
    "departure_date": "April 15",
    "arrival_city": "Athens",
    "arrival_airport": "Athens International Airport \"Eleftherios Venizelos\"",
    "arrival_time": "11:10 AM",
    "arrival_day": "Wednesday",
    "arrival_date": "April 16",
    "total_duration": "12 hr 10 min",
    "layovers": [
      {
        "layover_city": "Montreal",
        "layover_airport": "Montr\u00e9al-Pierre Elliott Trudeau International Airport",
        "layover_duration": "1 hr 47 min",
        "layover_count": 1
      }
    ]
  },
  {
    "price": 1210,
    "currency": "USD",
    "flight_type": "1 stop",
    "airline": "KLM",
    "departure_city": "Toronto",
    "departure_airport": "Pearson International Airport",
    "departure_time": "4:45 PM",
    "departure_day": "Tuesday",
    "departure_date": "April 15",
    "arrival_city": "Athens",
    "arrival_airport": "Athens International Airport \"Eleftherios Venizelos\"",
    "arrival_time": "12:40 PM",
    "arrival_day": "Wednesday",
    "arrival_date": "April 16",
    "total_duration": "12 hr 55 min",
    "layovers": [
      {
        "layover_city": "Amsterdam",
        "layover_airport": "Amsterdam Airport Schiphol",
        "layover_duration": "2 hr 25 min",
        "layover_count": 1
      }
    ]
  },
  {
    "price": 1203,
    "currency": "USD",
    "flight_type": "1 stop",
    "airline": "Air France",
    "departure_city": "Toronto",
    "departure_airport": "Pearson International Airport",
    "departure_time": "6:15 PM",
    "departure_day": "Tuesday",
    "departure_date": "April 15",
    "arrival_city": "Athens",
    "arrival_airport": "Athens International Airport \"Eleftherios Venizelos\"",
    "arrival_time": "2:20 PM",
    "arrival_day": "Wednesday",
    "arrival_date": "April 16",
    "total_duration": "13 hr 5 min",
    "layovers": [
      {
        "layover_city": "Paris",
        "layover_airport": "Paris Charles de Gaulle Airport",
        "layover_duration": "2 hr 25 min",
        "layover_count": 1
      }
    ]
  }
]

Selected Flight:
{
  "price": 1350,
  "currency": "USD",
  "flight_type": "1 stop",
  "airline": "Air France",
  "departure_city": "Toronto",
  "departure_airport": "Toronto Pearson International Airport",
  "departure_time": "11:00 PM",
  "departure_day": null,
  "departure_date": "April 15",
  "arrival_city": "Athens",
  "arrival_airport": "Athens International Airport",
  "arrival_time": "6:00 PM",
  "arrival_day": null,
  "arrival_date": "April 16",
  "total_duration": null,
  "layovers": [
    {
      "layover_city": "Paris",
      "layover_airport": "Charles de Gaulle Airport",
      "layover_duration": "1 hr 30 min",
      "layover_count": 1
    }
  ]
}

Would you say this flight satisfies the following criterion?

Criterion: premium economy

Respond yes, no, or unsure. Think step by step. Wrap your responses in the tags <think> </think> and <response> </response>.

<think>
All the flights provided in the search results (including the selected one) were returned using the search criterion of "premium economy" seating class. \
There is no explicit mention in the selected flight details, but we can rely on the initial search parameters provided \
to infer that this flight was selected from a list of premium economy options.
</think>

<response>
yes
</response>


Would you say this flight satisfies the following criterion?

Criterion: shortest layover

Respond yes, no, or unsure. Think step by step. Wrap your responses in the tags <think> </think> and <response> </response>.

<think>
To determine if the selected flight satisfies the criterion of having the shortest layover, I need to compare the layover durations of all the listed flights. Let's break it down step by step:

Layover durations for each flight:
Flight 1 (Selected Flight) – Layover in Paris: 1 hr 30 min
Flight 2 – Layover in Montreal: 1 hr 47 min
Flight 3 – Layover in Amsterdam: 2 hr 25 min
Flight 4 – Layover in Paris: 2 hr 25 min

Shortest Layover:
The selected flight has a layover duration of 1 hr 30 min, which is shorter than all the other flights' layovers.
</think>

<response>
yes
</response>


Example 2:

Date and Time: Mon, Oct 07, 2024 23:24:15

Search Criteria:
- Ticket Type: One way
- Seating Class: Business
- Origin: San Francisco
- Destination: Auckland
- Departure: Wed, Oct 16
- Return: None
- Sorting Criterion: None
- Num Passengers: 1 passenger

Top flights based on search criteria:
[
  {
    "price": 5585,
    "currency": "USD",
    "flight_type": "2 stops",
    "airlines": [
      "Alaska",
      "Fiji Airways"
    ],
    "operated_by": "SkyWest Airlines as Alaska SkyWest",
    "departure_city": "San Francisco",
    "departure_airport": "San Francisco International Airport",
    "departure_time": "7:57 PM",
    "departure_day": "Wednesday",
    "departure_date": "October 16",
    "arrival_city": "Auckland",
    "arrival_airport": "Auckland Airport",
    "arrival_time": "1:40 PM",
    "arrival_day": "Friday",
    "arrival_date": "October 18",
    "total_duration": "21 hr 43 min",
    "layovers": [
      {
        "layover_city": "Los Angeles",
        "layover_airport": "Los Angeles International Airport",
        "layover_duration": "2 hr 26 min",
        "layover_count": 1
      },
      {
        "layover_city": "Nadi",
        "layover_airport": "Nadi International Airport",
        "layover_duration": "3 hr 45 min",
        "layover_count": 2
      }
    ]
  },
  {
    "price": 7590,
    "currency": "USD",
    "flight_type": "1 stop",
    "airlines": [
      "United",
      "Qantas"
    ],
    "departure_city": "San Francisco",
    "departure_airport": "San Francisco International Airport",
    "departure_time": "11:35 PM",
    "departure_day": "Wednesday",
    "departure_date": "October 16",
    "arrival_city": "Auckland",
    "arrival_airport": "Auckland Airport",
    "arrival_time": "6:25 PM",
    "arrival_day": "Friday",
    "arrival_date": "October 18",
    "total_duration": "22 hr 50 min",
    "layovers": [
      {
        "layover_city": "Melbourne",
        "layover_airport": "Melbourne Airport",
        "layover_duration": "3 hr 35 min",
        "layover_count": 1
      }
    ]
  },
  {
    "price": 5753,
    "currency": "USD",
    "flight_type": "2 stops",
    "airlines": [
      "EVA Air",
      "Qantas"
    ],
    "departure_city": "San Francisco",
    "departure_airport": "San Francisco International Airport",
    "departure_time": "1:10 AM",
    "departure_day": "Wednesday",
    "departure_date": "October 16",
    "arrival_city": "Auckland",
    "arrival_airport": "Auckland Airport",
    "arrival_time": "3:20 PM",
    "arrival_day": "Friday",
    "arrival_date": "October 18",
    "total_duration": "42 hr 10 min",
    "layovers": [
      {
        "layover_city": "Taipei City",
        "layover_airport": "Taiwan Taoyuan International Airport",
        "layover_duration": "3 hr 30 min",
        "layover_count": 1
      },
      {
        "layover_city": "Brisbane",
        "layover_airport": "Brisbane Airport",
        "layover_duration": "13 hr 10 min",
        "layover_count": 2
      }
    ]
  },
  {
    "price": 7590,
    "currency": "USD",
    "flight_type": "1 stop",
    "airlines": [
      "United",
      "Qantas"
    ],
    "departure_city": "San Francisco",
    "departure_airport": "San Francisco International Airport",
    "departure_time": "11:30 PM",
    "departure_day": "Wednesday",
    "departure_date": "October 16",
    "arrival_city": "Auckland",
    "arrival_airport": "Auckland Airport",
    "arrival_time": "12:25 AM",
    "arrival_day": "Saturday",
    "arrival_date": "October 19",
    "total_duration": "28 hr 55 min",
    "layovers": [
      {
        "layover_city": "Brisbane",
        "layover_airport": "Brisbane Airport",
        "layover_duration": "11 hr 45 min",
        "layover_count": 1
      }
    ]
  },
  {
    "price": 7582,
    "currency": "USD",
    "flight_type": "1 stop",
    "airlines": [
      "United",
      "Air New Zealand"
    ],
    "departure_city": "San Francisco",
    "departure_airport": "San Francisco International Airport",
    "departure_time": "8:57 AM",
    "departure_day": "Wednesday",
    "departure_date": "October 16",
    "arrival_city": "Auckland",
    "arrival_airport": "Auckland Airport",
    "arrival_time": "6:00 AM",
    "arrival_day": "Friday",
    "arrival_date": "October 18",
    "total_duration": "25 hr 3 min",
    "layovers": [
      {
        "layover_city": "Los Angeles",
        "layover_airport": "Los Angeles International Airport",
        "layover_duration": "10 hr 30 min",
        "layover_count": 1
      }
    ]
  }
]

Selected Flight:
{
  "price": 5585,
  "currency": "USD",
  "flight_type": "2 stops",
  "airlines": [
    "Alaska",
    "Fiji Airways"
  ],
  "operated_by": "SkyWest Airlines as Alaska SkyWest",
  "departure_city": "San Francisco",
  "departure_airport": "San Francisco International Airport",
  "departure_time": "7:57 PM",
  "departure_day": "Wednesday",
  "departure_date": "October 16",
  "arrival_city": "Auckland",
  "arrival_airport": "Auckland Airport",
  "arrival_time": "1:40 PM",
  "arrival_day": "Friday",
  "arrival_date": "October 18",
  "total_duration": "21 hr 43 min",
  "layovers": [
    {
      "layover_city": "Los Angeles",
      "layover_airport": "Los Angeles International Airport",
      "layover_duration": "2 hr 26 min",
      "layover_count": 1
    },
    {
      "layover_city": "Nadi",
      "layover_airport": "Nadi International Airport",
      "layover_duration": "3 hr 45 min",
      "layover_count": 2
    }
  ]
}

Would you say this flight satisfies the following criterion?

Criterion: Boeing 787

Respond yes, no, or unsure. Think step by step. Wrap your responses in the tags <think> </think> and <response> </response>.

<think>
There is no mention in the provided information about the specific aircraft used on this flight. \
Since there is no direct reference to the aircraft type in the flight data, \
and without further information from the airlines or booking system about the planes being used, \
it is impossible to confirm that a Boeing 787 is being used on this route.
</think>

<response>
unsure
</response>
"""
