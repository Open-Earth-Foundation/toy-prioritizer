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
    print(top_actions)  # Debugging print statement
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["city", "action", "score"])
        writer.writeheader()
        for action in top_actions:
            writer.writerow(
                {
                    "city": action["city"]["name"],
                    "action": action["action"],
                    "score": action["score"],
                }
            )


def write_full_scores(output_file, all_scores):
    """Function to write all scores and LLM outputs to a CSV file."""
    with open(output_file, "w", newline="") as f:
        fieldnames = ["city", "action", "score", "llm_output"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in all_scores:
            writer.writerow(
                {
                    "city": entry["city"],
                    "action": entry["action"],
                    "score": entry["score"],
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
    print(f"Emissions High Degree: {emissions_high_degree}")
    print(f"Emissions Medium Degree: {emissions_medium_degree}")
    print(f"Emissions Low Degree: {emissions_low_degree}")
    print(f"Cost Low Degree: {cost_low_degree}")
    print(f"Cost Medium Degree: {cost_medium_degree}")
    print(f"Cost High Degree: {cost_high_degree}")
    print(f"Time Short Degree: {time_short_degree}")
    print(f"Time Medium Degree: {time_medium_degree}")
    print(f"Time Long Degree: {time_long_degree}")
    print(f"Risk High Degree: {risk_high_degree}")
    print(f"Risk Medium Degree: {risk_medium_degree}")
    print(f"Risk Low Degree: {risk_low_degree}")
    print(f"Environment Match Degree: {environment_match_degree}")
    print(f"Population Match Degree: {population_match_degree}")
    print("##############################################")
    print('')

    # Compute the average degree
    average_degree = sum(degrees) / len(degrees)

    # Now, map the average degree to a score between 0 and 100

    score = average_degree * 100

    return score


def quantitative_score(city, action):
    score = 0

    # Add score for emissions_reduction
    if action["emissions_reduction"] == "":
        score += 0
    else:
        action_emissions_reduction = action["emissions_reduction"]
        score += (
            min(action_emissions_reduction, MAX_EMISSIONS_REDUCTIONS)
            / MAX_EMISSIONS_REDUCTIONS
        ) * SCORE_MAX

    # Add score for risk_reduction
    if action["risk_reduction"] == "":
        score += 0
    else:
        score += scale_scores.get(action["risk_reduction"], 0) * SCORE_MAX

    # Add score for environment
    if action["environment"] == "":
        score += SCORE_MAX
    else:
        score += SCORE_MAX if (action["environment"] == city["environment"]) else 0.0

    # Add score for population
    if action["population"] == "" or city["population"] == "":
        score += SCORE_MAX / 2.0
    else:
        city_population = city["population"]
        action_population = action["population"]
        if action_population == city_population:
            score += SCORE_MAX
        else:
            diff = abs(action_population - city_population)
            if diff == 0:
                ratio = 1.0
            else:
                ratio = min(city_population / diff, 1.0)
            score += ratio * SCORE_MAX

    # Add score for time_in_years
    if action["time_in_years"] == "":
        score += 0
    else:
        score += (
            1 - (min(action["time_in_years"], MAX_TIME_IN_YEARS) / MAX_TIME_IN_YEARS)
        ) * SCORE_MAX

    # Add score for cost
    if city["budget"] == "" or action["cost"] == "":
        score += 0
    else:
        city_budget = city["budget"]
        action_cost = action["cost"]
        if city_budget == 0:
            ratio = 1.0  # Avoid division by zero
        else:
            ratio = min(action_cost, city_budget) / city_budget
        score += (1 - ratio) * SCORE_MAX

    return score


def qualitative_prioritizer(cities, actions, number_of_actions=5):
    top_actions = []
    all_scores = []  # List to store all scores
    for city in cities:
        scores = {}
        for action in actions:
            score, llm_response = qualitative_score(city, action)
            scores[action["name"]] = score
            all_scores.append(
                {
                    "city": city["name"],
                    "action": action["name"],
                    "score": score,
                    "llm_output": llm_response,
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
                }
                for action in top_action_names
            ]
        )
    return top_actions, all_scores  # Return both top actions and all scores


def quantitative_prioritizer(cities, actions, number_of_actions=5):
    top_actions = []
    all_scores = []  # List to store all scores
    for city in cities:
        scores = {}
        for action in actions:
            score = quantitative_score(city, action)
            scores[action["name"]] = score
            all_scores.append(
                {
                    "city": city["name"],
                    "action": action["name"],
                    "score": score,
                    "llm_output": "",  # No LLM output for quantitative method
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
                }
                for action in top_action_names
            ]
        )
    return top_actions, all_scores  # Return both top actions and all scores


def fuzzy_prioritizer(cities, actions, number_of_actions=5):
    top_actions = []
    all_scores = []  # List to store all scores
    for city in cities:
        scores = {}
        for action in actions:
            score = fuzzy_score(city, action)
            scores[action["name"]] = score
            all_scores.append(
                {
                    "city": city["name"],
                    "action": action["name"],
                    "score": score,
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
                }
                for action in top_action_names
            ]
        )
    return top_actions, all_scores  # Return both top actions and all scores


def main(city_file, action_file, output_file, method, number_of_actions):
    cities = read_cities(city_file)
    actions = read_actions(action_file)
    if method == 'quantitative':
        top_actions, all_scores = quantitative_prioritizer(
            cities, actions, number_of_actions
        )
        # Write the full list of quantitative scores
        write_full_scores(
            "quantitative_scores.csv",
            sorted(all_scores, key=lambda x: x["score"], reverse=True),
        )
    elif method == 'qualitative':
        top_actions, all_scores = qualitative_prioritizer(
            cities, actions, number_of_actions
        )
        # Write the full list of qualitative scores including LLM outputs
        write_full_scores(
            "qualitative_scores.csv",
            sorted(all_scores, key=lambda x: x["score"], reverse=True),
        )
    elif method == 'fuzzy':
        top_actions, all_scores = fuzzy_prioritizer(
            cities, actions, number_of_actions
        )
        # Write the full list of fuzzy scores
        write_full_scores(
            "fuzzy_scores.csv",
            sorted(all_scores, key=lambda x: x["score"], reverse=True),
        )
    write_output(output_file, top_actions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("city_file")  # First positional argument
    parser.add_argument("action_file")  # Second positional argument
    parser.add_argument("output_file")  # Third positional argument
    parser.add_argument(
        "--method", choices=['qualitative', 'quantitative', 'fuzzy'], default='qualitative'
    )  # Optional argument to select the method
    parser.add_argument(
        "number_of_actions", type=int
    )  # Fourth positional argument
    args = parser.parse_args()

    main(
        args.city_file,
        args.action_file,
        args.output_file,
        args.method,
        args.number_of_actions,
    )
