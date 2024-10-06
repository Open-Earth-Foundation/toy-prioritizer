import os
import csv
from dotenv import load_dotenv
# Import the fuzzylogic library
from fuzzylogic.classes import Domain
from fuzzylogic.functions import R, S, triangular
import argparse
import re

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# If you're using the OpenAI library, you might need to import it appropriately
# from openai import OpenAI

# client = OpenAI(api_key=api_key)


def send_to_llm(prompt):
    # Assuming you have the OpenAI client set up
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
            You are a climate action expert, tasked to prioritize climate actions for cities.
             
            These are the rules to use for prioritizing a climate action for a city:
             
            - Lower cost actions are better than higher cost actions.
            - High emissions reductions are better than low emissions reductions.
            - High risk reduction is better than low risk reduction.
            - Actions that match the environment are better than those that don't.
            - Actions that match the population are better than those that don't.
            - Actions that take less time are better than those that take more time.
            """,
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def read_cities(city_file):
    cities = []
    with open(city_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert appropriate fields to integers, if not blank
            for field in [
                "population",
                "area",
                "budget",
                "total_emission",
                "energy_emissions",
                "transportation_emissions",
                "waste_emissions",
            ]:
                if row[field]:
                    row[field] = int(row[field].replace(",", ""))
                else:
                    row[field] = ""
            cities.append(row)
    return cities


def read_actions(action_file):
    actions = []
    with open(action_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert appropriate fields to integers, if not blank
            for field in [
                "emissions_reduction",
                "population",
                "time_in_years",
                "cost",
            ]:
                if row[field]:
                    row[field] = int(row[field].replace(",", ""))
                else:
                    row[field] = ""
            actions.append(row)
    return actions


def write_output(output_file, top_actions):
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["city", "action", "score", "qualitative_score"])
        writer.writeheader()
        for action in top_actions:
            writer.writerow(
                {
                    "city": action["city"]["name"],
                    "action": action["action"],
                    "score": action["score"],
                    "qualitative_score": action["qualitative_score"],
                }
            )


def write_full_scores(output_file, all_scores):
    """Function to write all scores and LLM outputs to a CSV file."""
    with open(output_file, "w", newline="") as f:
        fieldnames = ["city", "action", "score", "qualitative_score", "llm_output"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in all_scores:
            writer.writerow(
                {
                    "city": entry["city"],
                    "action": entry["action"],
                    "score": entry["score"],
                    "qualitative_score": entry["qualitative_score"],
                    "llm_output": entry["llm_output"],
                }
            )


def qualitative_score(city, action):
    prompt = f"""
    According to the rules given, how would you prioritize the following action for the city with name {city["name"]},
    population {city["population"]}, area {city["area"]}, environment {city["environment"]}, budget {city["budget"]},
    total GHG emissions in CO2eq {city["total_emission"]}, energy {city["energy_emissions"]},
    transportation emissions {city["transportation_emissions"]}, waste emissions {city["waste_emissions"]}, and risk {city["risk"]}?

    Action: {action["name"]}, cost {action["cost"]}, GHG emissions reduction in CO2eq {action["emissions_reduction"]}, risk reduction {action["risk_reduction"]}, environment {action["environment"]}, population {action["population"]}, time {action["time_in_years"]}

    Please return a score from 0 to 100, where 0 is the worst possible action and 100 is the best possible action.

    Response format: [SCORE]
    """

    llm_response = send_to_llm(prompt)
    # Extract the score from the response
    print(llm_response)
    print("##############################################")
    match = re.search(r"\[(\d+)\]", llm_response)
    if match:
        score = int(match.group(1))
    else:
        print("No match found")
        score = 0  # Default score if no match is found
    print(score)
    return score, llm_response  # Return both score and LLM response


# Constants for quantitative scoring
SCORE_MAX = 100 / 6
MAX_EMISSIONS_REDUCTIONS = 500000
scale_scores = {
    "Very High": 1.0,
    "High": 0.75,
    "Medium": 0.5,
    "Low": 0.25,
    "Very Low": 0.0,
}
MAX_TIME_IN_YEARS = 20
MAX_COST = 60000000

# Define fuzzy domains and sets
# Importing the fuzzy logic library functions is already done at the top
# Define fuzzy domains and sets
# Define fuzzy domains and sets

# Emissions Reduction
EMISSIONS_REDUCTION = Domain(
    "Emissions Reduction", 0, MAX_EMISSIONS_REDUCTIONS, res=1000
)
EMISSIONS_REDUCTION.low = S(0, MAX_EMISSIONS_REDUCTIONS / 3)
EMISSIONS_REDUCTION.medium = triangular(
    MAX_EMISSIONS_REDUCTIONS / 3, MAX_EMISSIONS_REDUCTIONS * 2 / 3
)
EMISSIONS_REDUCTION.high = R(
    MAX_EMISSIONS_REDUCTIONS * 2 / 3, MAX_EMISSIONS_REDUCTIONS
)

# Cost
COST = Domain("Cost", 0, MAX_COST, res=1000000)
COST.low = R(0, MAX_COST / 3)  # Lower cost is better
COST.medium = triangular(MAX_COST / 3, MAX_COST * 2 / 3)
COST.high = S(MAX_COST * 2 / 3, MAX_COST)

# Time in years
TIME = Domain("Time", 0, MAX_TIME_IN_YEARS, res=1)
TIME.short = R(0, MAX_TIME_IN_YEARS / 2)  # Shorter time is better
TIME.medium = triangular(MAX_TIME_IN_YEARS / 3, MAX_TIME_IN_YEARS * 2 / 3)
TIME.long = S(MAX_TIME_IN_YEARS / 2, MAX_TIME_IN_YEARS)

# Risk Reduction
RISK_REDUCTION = Domain("Risk Reduction", 0, 1, res=0.01)
RISK_REDUCTION.low = S(0, 0.3)
RISK_REDUCTION.medium = triangular(0.3, 0.7)
RISK_REDUCTION.high = R(0.7, 1)

# Output Score
SCORE = Domain("Score", 0, 100, res=1)
SCORE.very_low = S(0, 20)
SCORE.low = triangular(10, 40)
SCORE.medium = triangular(30, 70)
SCORE.high = triangular(60, 90)
SCORE.very_high = R(80, 100)


def defuzzify_score(score):
    """
    Convert a numerical score into a qualitative value.
    """
    if score <= 20:
        return "Very Low"
    elif 20 < score <= 40:
        return "Low"
    elif 40 < score <= 60:
        return "Medium"
    elif 60 < score <= 80:
        return "High"
    else:
        return "Very High"


def fuzzy_score(city, action):
    # For each parameter, get the value and degrees of membership

    # Emissions Reduction
    if action["emissions_reduction"] == "":
        emissions_reduction_value = 0
    else:
        emissions_reduction_value = action["emissions_reduction"]

    emissions_high_degree = EMISSIONS_REDUCTION.high(emissions_reduction_value)
    emissions_medium_degree = EMISSIONS_REDUCTION.medium(emissions_reduction_value)
    emissions_low_degree = EMISSIONS_REDUCTION.low(emissions_reduction_value)

    # Cost
    if action["cost"] == "":
        cost_value = MAX_COST  # Assume maximum cost if missing
    else:
        cost_value = action["cost"]

    cost_low_degree = COST.low(cost_value)
    cost_medium_degree = COST.medium(cost_value)
    cost_high_degree = COST.high(cost_value)

    # Time in years
    if action["time_in_years"] == "":
        time_value = MAX_TIME_IN_YEARS  # Assume maximum time if missing
    else:
        time_value = action["time_in_years"]

    time_short_degree = TIME.short(time_value)
    time_medium_degree = TIME.medium(time_value)
    time_long_degree = TIME.long(time_value)

    # Risk Reduction
    if action["risk_reduction"] == "":
        risk_reduction_value = 0
    else:
        risk_reduction_value = scale_scores.get(action["risk_reduction"], 0)

    risk_high_degree = RISK_REDUCTION.high(risk_reduction_value)
    risk_medium_degree = RISK_REDUCTION.medium(risk_reduction_value)
    risk_low_degree = RISK_REDUCTION.low(risk_reduction_value)

    # Environment Match
    environment_match_degree = 1.0 if action["environment"] == city["environment"] else 0.0

    # Population Match
    if action["population"] == "" or city["population"] == "":
        population_match_degree = 0.5  # Assume partial match
    else:
        action_population = action["population"]
        city_population = city["population"]
        if action_population == city_population:
            population_match_degree = 1.0
        else:
            diff = abs(action_population - city_population)
            ratio = 1 - (diff / max(action_population, city_population))
            population_match_degree = max(0, min(ratio, 1.0))

    # Now combine the degrees to get a score
    # For example, we can average the degrees of the 'good' sets

    degrees = [
        emissions_high_degree,
        cost_low_degree,
        time_short_degree,
        risk_high_degree,
        environment_match_degree,
        population_match_degree,
    ]

    # Combined print statement for easier commenting
    print(f"Degrees:\n"
          f"Emissions High: {emissions_high_degree}, Medium: {emissions_medium_degree}, Low: {emissions_low_degree}\n"
          f"Cost Low: {cost_low_degree}, Medium: {cost_medium_degree}, High: {cost_high_degree}\n"
          f"Time Short: {time_short_degree}, Medium: {time_medium_degree}, Long: {time_long_degree}\n"
          f"Risk High: {risk_high_degree}, Medium: {risk_medium_degree}, Low: {risk_low_degree}\n"
          f"Environment Match: {environment_match_degree}\n"
          f"Population Match: {population_match_degree}\n"
          f"##############################################\n")

    # Compute the average degree
    average_degree = sum(degrees) / len(degrees)

    # Now, map the average degree to a score between 0 and 100

    score = average_degree * 100

    return score


def fuzzy_prioritizer(cities, actions, number_of_actions=5):
    top_actions = []
    all_scores = []  # List to store all scores
    for city in cities:
        scores = {}
        for action in actions:
            score = fuzzy_score(city, action)
            qualitative_value = defuzzify_score(score)  # Defuzzify the score
            scores[action["name"]] = score
            all_scores.append(
                {
                    "city": city["name"],
                    "action": action["name"],
                    "score": score,
                    "qualitative_score": qualitative_value,
                    "llm_output": "",  # No LLM output for fuzzy method
                }
            )
        actions_keys = scores.keys()
        actions_keys = sorted(actions_keys, key=lambda x: scores[x], reverse=True)
        top_action_names = actions_keys[:number_of_actions]
        top_actions.extend(
            [
                {
                    "city": city,
                    "action": action,
                    "score": scores[action],
                    "qualitative_score": defuzzify_score(scores[action]),  # Defuzzify score for output
                }
                for action in top_action_names
            ]
        )
    return top_actions, all_scores  # Return both top actions and all scores