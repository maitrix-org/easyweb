import argparse
import json
import multiprocessing
import os
import random
import sys
import time
from datetime import datetime
from glob import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from my_frontend import OpenDevinSession

# questions as of 7/15
# questions = [
#     'Using google, what is the difference between the ages of the current partners of Pete Buttigieg and Taylor Swift?',
#     'Using google, what is the sum of Lebron James’s career high basketball game point, Brazil’s Soccer World Cup wins, and Ma Long’s official ping-pong rating according to the ITTF?',
#     'Using google, what’s the difference in flight time in minutes between LA to San Francisco and LA to San Jose?',
#     'Using Google, what is the total number of Grammy awards won by Adele, the number of Oscars won by Leonardo DiCaprio, and the number of Nobel Prizes won by Marie Curie?',
#     'Using Google, what is the average salary of a software engineer in Silicon Valley, a lawyer in New York City, and a doctor in Los Angeles?',
#     'I have a budget of $3000 per month. Can you give me 3 good options for apartment rental in Hudson yards, nyc using google?',
#     'Find me a laptop under $1,000 that has at least 16GB RAM and a dedicated GPU.',
#     'I live at 4463 Oak Grove Dr, La Cañada Flintridge, CA 91011, can you find some Mediterranean restaurant within a 10 mile radius that has a rating above 4.0 stars with more than 500 user ratings?',
#     'I want to invest $10,000 in stocks. Can you recommend three stable companies to invest in based on current market trends?',
#     'I have $500 to spend on a weekend getaway. Can you suggest three destinations near San Francisco with accommodation and activities within this budget?',
#     'Can you go on amazon and help me put a gaming computer of any kind into my shopping cart?',
#     'Help me find a table at fogo de chão in San Jose for 2 people using google.',
#     'Using google, can you purchase a One-way flight from Los Angeles to Tokyo this Sunday?',
#     'Help me reserve a hotel room in New York City for three nights starting next Friday using Google.',
#     'Using Google, can you purchase tickets to the next NBA game in Los Angeles?',
# ]
# questions = [
#     'Using google, what is the difference between the ages of Pete Buttigieg’s husband and Taylor Swift’s current boyfriend?',
#     'Using google, what is the sum of Lebron James’s career high basketball game point, Brazil’s Soccer World Cup wins, and Ma Long’s official ping-pong rating according to the ITTF?',
#     'Using google, what’s the difference in flight time in minutes between LA to San Francisco and LA to San Jose?',
#     'Using Google, what is the sum of the years of the last Grammy award won by Adele, the last Oscar won by Leonardo DiCaprio, and last Nobel Prize won by Marie Curie?',
#     'Using Google, who has the highest salary? An average software engineer in Silicon Valley, an average lawyer in New York City, and an average doctor in Los Angeles?',
#     'Using Apartments.com Can you give me 3 options for apartment rentals in hudson yards, nyc? I have a budget of $3000 and prefer to have a river view.',
#     'Find me 3 options for laptops under $1,000 that have at least 16GB RAM and a dedicated GPU.',
#     'I live at 4463 Oak Grove Dr, La Cañada Flintridge, CA 91011, can you find 3 Mediterranean restaurants close by that have ratings above 4.0 stars and a large number of ratings?',
#     'I have $500 to spend on a weekend getaway. Can you suggest 3 destinations near san diego with accommodation and activities within this budget?',
#     # 'I want to buy a black mattress. Can you look at 3 e-commerce platforms and give me one good option from each?',
#     # 'I want to buy a black mattress. Can you look at Amazon, eBay, and Bed Bath Beyond and give me one good option from each?',
#     'I want to buy a black mattress. Can you look at Amazon and eBay, and give me one good option from each?',
#     'Can you go on amazon and help me put a gaming computer of any kind into my shopping cart?',
#     'Help me find a table at fogo de chão in San Jose for 2 people using google. Do not use the official website',
#     'Using google, can you purchase a one way flight from Los Angeles to Tokyo this Sunday using google flight?',
#     'Help me reserve a hotel room in New York City for three nights starting this Sunday using Google.',
#     'Using Google, can you purchase tickets to the next Taylor Swift concert close to me? My zipcode is 92093',
# ]
# 2024-07-23-18-14-41
# questions = [
#     "Using google search, what is the age difference between Pete Buttigieg's current partner and Taylor Swift's current partner?",
#     'Using google search, what is the difference in flight time in minutes between LA to San Francisco and LA to San Jose?',
#     'Using google search, who has the highest salary? An average software engineer in Silicon Valley, an average lawyer in New York City, or an average doctor in Los Angeles?',
#     'What is the top-3 best-selling women’s dress on Amazon?',
#     'Compare the difference in time for walking and driving route from Carnegie Mellon University to AMC Waterfront.',
#     'Tell me the number of reviews that the Kebab Shop next to UCSD received on Yelp that mention the term "lamb"',
#     'Can you find three laptops from Amazon and three from eBay, each priced under $1,000, with at least 16GB of RAM and a dedicated GPU?',
#     'I live at La Cañada Flintridge, CA. Can you find three nearby Mediterranean restaurants with ratings above 4.0 stars and reviews that mention lamb using Yelp?',
#     'I want to buy a black queen mattress with a budget of $300. Can you find one option each from Amazon and eBay?',
#     'On Amazon, find the cheapest gaming computer with a rating higher than 4.0 and add it to the shopping cart.',
#     'Can you purchase a one way flight from Los Angeles to San Francisco this Saturday using google flight?',
#     'Can you find two cheapest available tickets to the next Taylor Swift concert?',
#     "Use google search as calculator to answer the following question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
#     'Use google search as calculator to answer the following question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?',
#     'Use google search as calculator to answer the following question: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?',
# ]

# questions = [
#     'Go to Ryanair and find top-rated outdoor activity events in Spain that are happening on May 1, then book the tickets for two adults in the morning group with an English guide.',
#     'Go to SpotHero and sell my parking space in Houston. Name: James Smith, Email: buckeye.foobar@gmail.com, Phone: 8888888888. Address: 123rd St.',
#     'Go to Expedia and find first class flights for May 27 from JFK, New York to Heathrow, London for 2 adults, one 4-year-old child, and one infant under 1 year old who will sit on a lap.',
#     'Go to BookDepository and browse fiction audio books sorted by lowest price.',
#     'Go to foxsports and find the odds for upcoming NHL matches.',
#     'Go to CVS and browse cough medicine that is rated 4 stars and above and is $15-$20.',
#     'Go to boardgamegeek and buy the number 4 ranked board game on the geekmarket.',
#     'Go to Target and find multi-colored fillable eggs for Easter made from polypropylene priced between 5 to 10 dollars for pick-up from Barboursville, zip 25504.',
#     'Go to aa and find the lowest fare for JFK, NY to Heathrow, London and nearby airports for 1 adult on April 22, one way.',
#     'Go to cargurus and find an offer for a black Honda with VIN number 1HGCM66543A064159 with 155,000 mileage near 49102. The car should be in good condition with no damage, 2+ keys, and paid off.',
#     'Go to resy and find the top restaurants in Boston to reserve for April 22.',
#     'Go to eBay and get a Hasbro Hulk action figure manufactured in 1990 with the lowest price plus shipping.',
#     'Go to rentalcars and check prices for luxury sedan cars in Houston with insurance.',
#     'Go to kbb and check reviews for the best electric SUV, then find the 1-star rated review and mark it helpful.',
#     'Go to last.fm and find events starting from April 1, 2023, within 100 miles of New York. Check who is attending the top listed event and follow all of them.',
#     'Go to sports.yahoo and find the results of the most recent NFL games.',
#     'Go to Discogs and find the address and store hours for the Armageddon Shop record store in Boston.',
#     'Go to qatarairways and retrieve booking for 123456 with last name Smith.',
#     'Go to IKEA and open the design tool for a new home office.',
#     'Go to Yelp and find parking in California city for limos that also offers military discounts and free Wi-Fi.',
# ]

gaia_web_questions = [
    'I’m researching species that became invasive after people who kept them as pets released them. There’s a certain species of fish that was popularized as a pet by being the main character of the movie Finding Nemo. According to the USGS, where was this fish found as a nonnative species, before the year 2020? I need the answer formatted as the five-digit zip codes of the places the species was found, separated by commas if there is more than one place.',
    'If we assume all articles published by Nature in 2020 (articles, only, not book reviews/columns, etc) relied on statistical significance to justify their findings and they on average came to a p-value of 0.04, how many papers would be incorrect as to their claims of statistical significance? Round the value up to the next integer.',
    'If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary.',
    'How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.',
    "The object in the British Museum's collection with a museum number of 2012,5015.17 is the shell of a particular mollusk species. According to the abstract of a research article published in Science Advances in 2021, beads made from the shells of this species were found that are at least how many thousands of years old?",
    'According to github, when was Regression added to the oldest closed numpy.polynomial issue that has the Regression label in MM/DD/YY?',
    'What are the EC numbers of the two most commonly used chemicals for the virus testing method in the paper about SPFMV and SPCSV in the Pearl Of Africa from 2016? Return the semicolon-separated numbers in the order of the alphabetized chemicals.',
    'In April of 1977, who was the Prime Minister of the first place mentioned by name in the Book of Esther (in the New International Version)?',
    "Use density measures from the chemistry materials licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials as compiled 08/21/2023.\n\nI have a gallon of honey and a gallon of mayonnaise at 25C. I remove one cup of honey at a time from the gallon of honey. How many times will I need to remove a cup to have the honey weigh less than the mayonaise? Assume the containers themselves weigh the same.",
    'Of the authors (First M. Last) that worked on the paper "Pie Menus or Linear Menus, Which Is Better?" in 2015, what was the title of the first paper authored by the one that had authored prior papers?',
    'In the NCATS PubChem compound database for Food Additive Status classification, find the compound that has a molecular weight of 100 g/mol or less, 6 heavy atoms, 1 or fewer hydrogen bond acceptors, and a complexity between 10 and 15. Of the shared gene-chemical co-occurrences between its two possible enzyme transformations, what is the PubChem CID of the heaviest by molecular weight?',
    'I need to fact-check a citation. This is the citation from the bibliography:\n\nGreetham, David. "Uncoupled: OR, How I Lost My Author(s)." Textual Cultures: Texts, Contexts, Interpretation, vol. 3 no. 1, 2008, p. 45-46. Project MUSE, doi:10.2979/tex.2008.3.1.44.\n\nAnd this is the in-line citation:\n\nOur relationship with the authors of the works we read can often be “obscured not by a "cloak of print" but by the veil of scribal confusion and mis-transmission” (Greetham 45-46).\n\nDoes the quoted text match what is actually in the article? If Yes, answer Yes, otherwise, give me the word in my citation that does not match with the correct one (without any article).',
    'Which contributor to the version of OpenCV where support was added for the Mask-RCNN model has the same name as a former Chinese head of government when the names are transliterated to the Latin alphabet?',
    "What two-word type of model did Manash Pratim Kashyap's and PS Fader's studies in customer retention studies published during 2018-2019 have in common (no punctuation)?",
    'How many High Energy Physics - Lattice articles listed in January 2020 on Arxiv had ps versions available?',
    'What is the minimum number of page links a person must click on to go from the english Wikipedia page on The Lord of the Rings (the book) to the english Wikipedia page on A Song of Ice and Fire (the book series)? In your count, include each link you would click on to get to the page. Use the pages as they appeared at the end of the day on July 3, 2023.',
    'I went to Virtue restaurant & bar in Chicago for my birthday on March 22, 2021 and the main course I had was delicious!  Unfortunately, when I went back about a month later on April 21, it was no longer on the dinner menu.  Using the Wayback Machine, can you help me figure out which main course was on the dinner menu for Virtue on March 22, 2021 but not April 21, 2021? Answer using the singular form, without articles.',
    "In Emily Midkiff's June 2014 article in a journal named for the one of Hreidmar's sons that guarded his house, what word was quoted from two different authors in distaste for the nature of dragon depictions?",
    "It is 1999. Before you party like it is 1999, please assist me in settling a bet.\n\nFiona Apple and Paula Cole released albums prior to 1999. Of these albums, which didn't receive a letter grade from Robert Christgau? Provide your answer as a comma delimited list of album titles, sorted alphabetically.",
    "Under DDC 633 on Bielefeld University Library's BASE, as of 2020, from what country was the unknown language article with a flag unique from the others?",
    'Compute the check digit the Tropicos ID for the Order Helotiales would have if it were an ISBN-10 number.',
    "The Metropolitan Museum of Art has a portrait in its collection with an accession number of 29.100.5. Of the consecrators and co-consecrators of this portrait's subject as a bishop, what is the name of the one who never became pope?",
    "In Nature journal's Scientific Reports conference proceedings from 2012, in the article that did not mention plasmons or plasmonics, what nano-compound is studied? Don't use the prefix nano in your answer if there is one.",
    'According to Google Finance, when was the first year the Apple stock went above $50 (without adjusting for stock split)?',
    "According to Box Office Mojo's 2020 Worldwide Box Office list, how many of the top 10 highest-grossing worldwide movies are also on the top 10 highest-grossing domestic movies? Your answer should be a numerical integer value.",
    'In the year 2022, and before December, what does "R" stand for in the three core policies of the type of content that was violated in the public logs on the Legume Wikipedia page?',
    'Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?',
    'On a leap day before the year 2008, a joke was removed from the Wikipedia page for “Dragon”. What was the phrase that was removed? Give the phrase as it appeared on the page, but without punctuation.',
    "What is the volume in milliliters of a system comprised of 0.312 kg Freon-12 refrigerant when placed at the bottom of the Marianas Trench and allowed to stabilize at the Trench's peak temperature, rounded to the nearest mL? Provide your answer as just an integer value.",
    'The Latin root of the Yola word "gimlie" shares a spelling with a Spanish word. What is the Google translation of the source title for the 1994 example sentence for that word in the Collins Spanish-to-English dictionary online? Answer in plain text, without punctuation.',
    "Find the value of x to the nearest tenth: Lx = (d/dx * (A * x-squared)) + 4-thousand'n'ninety-7 minus C\nWhere L is the last two digits of the year of the Venezuelan Declaration of Independence,\nA is the number of colors in the TikTok logo as of July 2023, excluding black and white,\nand C is the height of the average woman in the Philippines according to a July 2023 Business Insider article, rounded to the nearest whole centimeter",
    'On July 15, 2008, Phys.org published an article about a catastrophe. Find the explosive force of this catastrophe according to Encyclopedia Britannica, then find the name of the US nuclear test that had the same yield. Your answer should only be the last word of the name of the test.',
    'How many edits were made to the Wikipedia page on Antidisestablishmentarianism from its inception until June of 2023?',
    'How many nonindigenous crocodiles were found in Florida from the year 2000 through 2020? You can get the data from the USGS Nonindigenous Aquatic Species database.',
    "The work referenced in footnote 397 of Federico Lauria's 2014 dissertation is also the source for the titles of two paintings in the Smithsonian American Art Museum's collection, as of August 2023. What is the absolute difference between the chapter numbers of the chapters that the titles of these two paintings quote?",
    'As of the 2020 census, what was the population difference between the largest county seat and smallest county seat, by land area of the county seat, in Washington state? For population figures, please use the official data from data.census.gov. Please report the integer difference.',
    'This is a secret message my friend gave me. It says where we should meet for our picnic on Friday. The only problem is, it’s encrypted in the Caesar cipher, so I can’t read it. Can you tell me what it says? This is the message:\n\nZsmxsm sc sx Zyvilsec Zvkjk.',
    'Who composed the song that was performed by a rooster and a hamster in separate animated videos at separate tempos with different lyrics? Answer using the format First name Last name.',
    "What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials as compiled 08/21/2023?",
    'I’m thinking about selling my home, so I want to learn more about how homes in my area sold recently. I live in Pearl City, Hawaii, which is on the island of Oahu. I know two homes near me that sold in 2022 were 2072 Akaikai Loop, and 2017 Komo Mai Drive. Find which of those homes sold for more in 2022, and tell me how much it sold for. Don’t put commas or decimal places in the answer.',
    'How many times was a Twitter/X post cited as a reference on the english Wikipedia pages for each day of August in the last June 2023 versions of the pages?',
    'On ScienceDirect, what is the difference to 3 decimal places in the sample standard deviations of the number of Reference Works in each Life Science domain compared to Health Sciences as of 2022?',
    "What is the last word before the second chorus of the King of Pop's fifth single from his sixth studio album?",
    'In the Scikit-Learn July 2017 changelog, what other predictor base command received a bug fix? Just give the name, not a path.',
    "It's May 2023, and I'm about to drive across the U.S. from California to Maine. I always recycle my water bottles at the end of a trip, and I drink 5 12-ounce water bottles for every 100 miles I travel, rounded to the nearest 100. Assuming I follow I-40 from Los Angeles to Cincinnati, then take I-90 from Cincinnati to Augusta, how many dollars will I get back according to Wikipedia?",
    'The longest-lived vertebrate is named after an island.  According to Wikipedia as of January 1, 2021, what is the 2020 estimated population of that island, to the nearest thousand?',
    'On the DeepFruits fruit detection graph on Connected Papers from 2016, what feature caused the largest bubble to be the size it is?',
    'During the first week of August 2015, one of the NASA Astronomy Pictures of the Day shows the lights of a city on the horizon. The namesake of this city also has a landmark building in Chicago named after him. What is the name of the architectural firm that designed this landmark building? Give the first name appearing in the name of the firm as of June 2023.',
    'How many more blocks (also denoted as layers) in BERT base encoder than the encoder from the architecture proposed in Attention is All You Need?',
    "All of the individuals who formally held the position of United States secretary of homeland security prior to April 2019, excluding those who held the position in an acting capacity, have a bachelor's degree. Of the universities that these bachelor's degrees were from, which is the westernmost university and which is the easternmost university? Give them to me as a comma-separated list, I only want the name of the cities where the universities are located, with the westernmost city listed first.",
    'On Cornell Law School website\'s legal information institute, under the fifth section of federal rules alphabetically, what word was deleted in the last amendment to the first rule in the article that has "witnesses" in the most titles as of 2021?',
    'Of the cities within the United States where U.S. presidents were born, which two are the farthest apart from the westernmost to the easternmost going east, giving the city names only? Give them to me in alphabetical order, in a comma-separated list',
    'According to Girls Who Code, how long did it take in years for the percentage of computer scientists that were women to change by 13% from a starting point of 37%?',
    'What was the complete title of the book in which two James Beard Award winners recommended the restaurant where Ali Khan enjoyed a New Mexican staple in his cost-conscious TV show that started in 2015? Write the numbers in plain text if there are some in the title.',
    'As of August 2023, who is the only winner of the US version of Survivor to be born in the month of May?',
    'How many at bats did the Yankee with the most walks in the 1977 regular season have that same season?',
    'In Audre Lorde’s poem “Father Son and Holy Ghost”, what is the number of the stanza in which some lines are indented?',
    "I'm curious about how much information is available for popular video games before their release. Find the Wikipedia page for the 2019 game that won the British Academy Games Awards. How many revisions did that page have before the month listed as the game's release date on that Wikipedia page (as of the most recent entry from 2022)?",
    'What is the absolute difference in tens of thousands between the population of chinstrap penguins on the Wikipedia page for penguin species populations as of the end of 2018 and the population recorded in the Nature.com "global population assessment of the Chinstrap penguin" article from 2020, assuming two penguins per breeding pair?',
    'A 5-man group made up of one tank, one healer, and three DPS is doing a dungeon that was just released in World of Warcraft. Two are plate wearers and two are cloth wearers. At the final boss, both the tank and the healer are casting holy spells. Ice and fire are being used, each one by a different DPS. A bear from the group is attacking the boss. Metamorphosis is cast. The Kilt of the Forgotten One drops as loot, but no one can use it. If all classes were using their class abilities and all classes are unique, what are the five classes in the group in alphabetical order separated by commas?',
    'On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. This article mentions a team that produced a paper about their observations, linked at the bottom of the article. Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?',
    'According to Openreview.net, at the NeurIPS 2022 Conference, how many papers by an author named Yuri were accepted with a "certain" recommendation?',
    'What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris patients from Jan-May 2018 as listed on the NIH website?',
    "I'd like to learn more about some popular reality television competition shows. As of the end of the 44th season of the American version of Survivor, how many more unique winners have there been compared to the number of winners of American Idol?",
    "Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited? Just give me the city name without abbreviations.",
    'As of May 2023, how many stops are between South Station and Windsor Gardens on MBTA’s Franklin-Foxboro line (not included)?',
]

gaia_system_prompt = """Finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""

gaia_web_questions = [q + '\n\n' + gaia_system_prompt for q in gaia_web_questions]

questions = gaia_web_questions

# math_questions = ["Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
#                   "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
#                   "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",]

# math_questions = ["Use google search as calculator to answer the following question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
#                   "Use google search as calculator to answer the following question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
#                   "Use google search as calculator to answer the following question: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",]

# math_questions = ["Use google search as calculator to answer the following question: Mark's car breaks down and he needs to get a new radiator. The cost for a new radiator is $400 but he goes to get it at a junk shop and gets it for 80% off. He then hires a mechanic to install it and it takes 3 hours at $50 an hour. How much did he pay?",
#                   "Use google search as calculator to answer the following question: Farmer Brown has 20 animals on his farm, all either chickens or cows. They have a total of 70 legs, all together. How many of the animals are chickens?",
#                   "Use google search as calculator to answer the following question: Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?",]

# math_questions = [
#     "Use google search as calculator to answer the following question: Mark's car breaks down and he needs to get a new radiator. The cost for a new radiator is $400 but he goes to get it at a junk shop and gets it for 80% off. He then hires a mechanic to install it and it takes 3 hours at $50 an hour. How much did he pay? Do the calculation on your own and don't search for the answer directly. Don't include extra text when you perform the calculation.",
#     "Use google search as calculator to answer the following question: Farmer Brown has 20 animals on his farm, all either chickens or cows. They have a total of 70 legs, all together. How many of the animals are chickens? Do the calculation on your own and don't search for the answer directly. Don't include extra text when you perform the calculation.",
#     "Use google search as calculator to answer the following question: Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have? Do the calculation on your own and don't search for the answer directly. Don't include extra text when you perform the calculation.",
# ]


# questions = math_questions
# finished_qids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
#                  26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
#                  52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

# finished_qids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
#                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
#                  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
#                  59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
#                  104, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
#                  171, 172, 182, 183, 184, 185, 186, 187, 188, 189, 190, 208, 209, 210, 211, 212,
#                  213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 234,
#                  235, 236, 237, 238, 239, 240, 241]

# finished_qids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
#                  20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
#                  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
#                  60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
#                  80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
#                  100, 104, 105, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
#                  143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
#                  159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174,
#                  175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
#                  208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
#                  224, 225, 226, 227, 228, 229, 234, 235, 236, 237, 238, 239, 240, 241]

# finished_qids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
#                  22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
#                  42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
#                  62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
#                  82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
#                  102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 130, 131, 132, 133, 134, 135,
#                  136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
#                  152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
#                  168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
#                  184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 208, 209, 210, 211,
#                  212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
#                  228, 229, 230, 231, 234, 235, 236, 237, 238, 239, 240, 241]

# finished_qids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
#                  20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
#                  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
#                  60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
#                  80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
#                  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
#                  115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
#                  130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
#                  145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
#                  160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174,
#                  175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
#                  190, 191, 192, 193, 194, 195, 196, 208, 209, 210, 211, 212, 213, 214, 215,
#                  216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
#                  231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
#                  246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256]

# finished_qids = [12, 13, 15, 16, 0, 1, 3, 4, 5, 6]
# finished_qids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19]
finished_qids = []
finished_questions = [
    'Can you find a one-way flight from New York to London departing next Friday?',
    'Can you find a round-trip, non-stop business class flight from Chicago to Tokyo, departing on October 20th and returning on October 27th?',
]


def run_question(args, qs, qid, start_datetime):
    # if qid < 9 or qid > 11:
    # if qid < 10:
    #     return
    if qid in finished_qids or qs[qid] in finished_questions:
        return

    if glob(f'my_evaluator_logs/*_{args.job_name}_{qid}_steps.json') != []:
        print(f'Existing log detected for question {qid}, skipping...')
        return

    random.seed(qid)
    question = qs[qid]
    session = OpenDevinSession(
        agent=args.agent, port=args.port, model=args.model, api_key=args.api_key
    )

    for agent_state in session.initialize(as_generator=True):
        print(qid, agent_state)

    action_messages = []
    max_steps = 30
    for message in session.run(question):
        if len(session.action_messages) > len(action_messages):
            diff = len(session.action_messages) - len(action_messages)
            new_action_messages = session.action_messages[-diff:]
            new_action_message = ';'.join(new_action_messages)
            action_messages += new_action_messages
            print(qid, new_action_message)
        if len(action_messages) >= max_steps:
            break

    os.makedirs('my_evaluator_logs', exist_ok=True)

    output_path = f'my_evaluator_logs/{start_datetime}_{args.job_name}_{qid}_steps.json'
    print('Saving log to', output_path)
    json.dump(session.raw_messages, open(output_path, 'w'))

    # session._close()

    time2sleep = 15 + random.random() * 15
    print(f'Sleeping for {time2sleep:.2f} seconds')
    time.sleep(time2sleep)


def main(qs):
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run evaluations at scale as if you're using the frontend"
    )

    # Add arguments
    parser.add_argument('job_name', type=str)
    parser.add_argument('--agent', type=str, default='WorldModelAgent')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--model', type=str, default='Meta-Llama-3.1-70B-Instruct')
    parser.add_argument('--api_key', type=str, default='token-abc123')
    parser.add_argument('--n_processes', type=int, default=3)

    # Parse the arguments
    args = parser.parse_args()

    start_datetime = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    # args_list = [(args, qid, start_datetime) for qid in range(len(questions)) if qid in [9, 13, 14]]
    args_list = [(args, qs, qid, start_datetime) for qid in range(len(qs))]
    # args_list = [(args, qid, start_datetime) for qid in range(7, len(questions))]
    with multiprocessing.Pool(processes=args.n_processes) as pool:
        pool.starmap(run_question, args_list)

    # run_question(args, 2, start_datetime)


if __name__ == '__main__':
    main(questions)
