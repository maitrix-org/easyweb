import argparse
import json
import multiprocessing
import os
import random
import time
from datetime import datetime

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

questions = [
    'Go to Ryanair and find top-rated outdoor activity events in Spain that are happening on May 1, then book the tickets for two adults in the morning group with an English guide.',
    'Go to SpotHero and sell my parking space in Houston. Name: James Smith, Email: buckeye.foobar@gmail.com, Phone: 8888888888. Address: 123rd St.',
    'Go to Expedia and find first class flights for May 27 from JFK, New York to Heathrow, London for 2 adults, one 4-year-old child, and one infant under 1 year old who will sit on a lap.',
    'Go to BookDepository and browse fiction audio books sorted by lowest price.',
    'Go to foxsports and find the odds for upcoming NHL matches.',
    'Go to CVS and browse cough medicine that is rated 4 stars and above and is $15-$20.',
    'Go to boardgamegeek and buy the number 4 ranked board game on the geekmarket.',
    'Go to Target and find multi-colored fillable eggs for Easter made from polypropylene priced between 5 to 10 dollars for pick-up from Barboursville, zip 25504.',
    'Go to aa and find the lowest fare for JFK, NY to Heathrow, London and nearby airports for 1 adult on April 22, one way.',
    'Go to cargurus and find an offer for a black Honda with VIN number 1HGCM66543A064159 with 155,000 mileage near 49102. The car should be in good condition with no damage, 2+ keys, and paid off.',
    'Go to resy and find the top restaurants in Boston to reserve for April 22.',
    'Go to eBay and get a Hasbro Hulk action figure manufactured in 1990 with the lowest price plus shipping.',
    'Go to rentalcars and check prices for luxury sedan cars in Houston with insurance.',
    'Go to kbb and check reviews for the best electric SUV, then find the 1-star rated review and mark it helpful.',
    'Go to last.fm and find events starting from April 1, 2023, within 100 miles of New York. Check who is attending the top listed event and follow all of them.',
    'Go to sports.yahoo and find the results of the most recent NFL games.',
    'Go to Discogs and find the address and store hours for the Armageddon Shop record store in Boston.',
    'Go to qatarairways and retrieve booking for 123456 with last name Smith.',
    'Go to IKEA and open the design tool for a new home office.',
    'Go to Yelp and find parking in California city for limos that also offers military discounts and free Wi-Fi.',
]


# math_questions = ["Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
#                   "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
#                   "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",]

# math_questions = ["Use google search as calculator to answer the following question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
#                   "Use google search as calculator to answer the following question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
#                   "Use google search as calculator to answer the following question: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",]

# math_questions = ["Use google search as calculator to answer the following question: Mark's car breaks down and he needs to get a new radiator. The cost for a new radiator is $400 but he goes to get it at a junk shop and gets it for 80% off. He then hires a mechanic to install it and it takes 3 hours at $50 an hour. How much did he pay?",
#                   "Use google search as calculator to answer the following question: Farmer Brown has 20 animals on his farm, all either chickens or cows. They have a total of 70 legs, all together. How many of the animals are chickens?",
#                   "Use google search as calculator to answer the following question: Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?",]

math_questions = [
    "Use google search as calculator to answer the following question: Mark's car breaks down and he needs to get a new radiator. The cost for a new radiator is $400 but he goes to get it at a junk shop and gets it for 80% off. He then hires a mechanic to install it and it takes 3 hours at $50 an hour. How much did he pay? Do the calculation on your own and don't search for the answer directly. Don't include extra text when you perform the calculation.",
    "Use google search as calculator to answer the following question: Farmer Brown has 20 animals on his farm, all either chickens or cows. They have a total of 70 legs, all together. How many of the animals are chickens? Do the calculation on your own and don't search for the answer directly. Don't include extra text when you perform the calculation.",
    "Use google search as calculator to answer the following question: Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have? Do the calculation on your own and don't search for the answer directly. Don't include extra text when you perform the calculation.",
]


# questions = math_questions


def run_question(args, qid, start_datetime):
    # if qid < 9 or qid > 11:
    #     return

    random.seed(qid)
    question = questions[qid]
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


def main():
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

    # Parse the arguments
    args = parser.parse_args()

    start_datetime = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    # args_list = [(args, qid, start_datetime) for qid in range(len(questions)) if qid in [9, 13, 14]]
    args_list = [(args, qid, start_datetime) for qid in range(len(questions))]
    # args_list = [(args, qid, start_datetime) for qid in range(7, len(questions))]
    with multiprocessing.Pool(processes=3) as pool:
        pool.starmap(run_question, args_list)

    # run_question(args, 2, start_datetime)


if __name__ == '__main__':
    main()
